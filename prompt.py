#!/usr/bin/env python3
"""
prompt.py
=========
Interactive text-generation script for the Mamba-3 FPGA accelerator on the
ARM Kria board.

Usage:
    python3 prompt.py

The script:
  1. Loads model weights from ../weights.pt if the file exists; otherwise
     initialises random weights (useful for smoke-testing on the board).
  2. Asks the user to type a prompt.
  3. Tokenises the prompt with a byte-level (256-token) vocabulary.
  4. Feeds the prompt tokens through the model to warm up the SSM state.
  5. Generates up to MAX_NEW_TOKENS=256 tokens in NSTEPS=64 batches, calling
     the FPGA accelerator for each batch via the CiFra OpenBus character
     devices.  Falls back to a pure-PyTorch reference implementation when
     the device files are absent (e.g. during development on a workstation).

Hardware interface (matches src/kria/host.cpp and src/hls/mamba3_step.cpp):
  Write device : /dev/cifra_openbus_write_32  (fp32 words, big send buffer)
  Read  device : /dev/cifra_openbus_read_32   (fp32 words, big recv buffer)

  Stream layout per DUT invocation (NSTEPS steps batched together):
    [CONSTANTS]  A  D  xproj  outproj  zproj
    [STATE_IN]   state
    [STEP × NSTEPS]  Bstate  Xstate  B  C  x  dt  trap  z
    ---- FPGA returns ----
    [OUT × NSTEPS]   out_buf  (one per step)
    [STATE_OUT]      state_out
"""

import os
import sys
import struct
import threading
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model configuration (must match src/hls/model_config.h)
# ---------------------------------------------------------------------------
BATCH  = 1
NHEADS = 4
HDIM   = 64
DSTATE = 32
MIMO   = 2
NSTEPS = 64          # steps batched per FPGA invocation (matches README)

# Derived sizes (element counts, not bytes)
AH_SIZE    = BATCH * NHEADS                       # 4
D_VEC_SIZE = NHEADS                               # 4
PROJ_SIZE  = MIMO * NHEADS * HDIM                 # 512
STATE_SIZE = BATCH * NHEADS * HDIM * DSTATE       # 8192
BC_SIZE    = BATCH * MIMO * NHEADS * DSTATE       # 256
X_SIZE     = BATCH * NHEADS * HDIM               # 256

D_MODEL  = NHEADS * HDIM   # 256  (the "hidden dimension" seen by the LM)
VOCAB    = 256             # byte-level vocabulary
MAX_NEW_TOKENS = 256

FPGA_WRITE_DEV = "/dev/cifra_openbus_write_32"
FPGA_READ_DEV  = "/dev/cifra_openbus_read_32"

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights.pt")


# ---------------------------------------------------------------------------
# Byte-level tokeniser
# ---------------------------------------------------------------------------

def encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class Mamba3LM(nn.Module):
    """
    Single-layer Mamba-3 language model whose SSM kernel is offloaded to the
    FPGA accelerator.

    Weights that are *constant* across autoregressive steps (A, D_vec, xproj,
    outproj, zproj) are stored as nn.Parameters and are streamed once per
    NSTEPS-step FPGA invocation.

    Per-step inputs (x, z, B, C, dt, trap, Bstate, Xstate) are derived from
    the token embedding via a linear in_proj.
    """

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.vocab_size = vocab_size

        # --- Embedding & output ---
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.lm_head   = nn.Linear(D_MODEL, vocab_size, bias=False)

        # --- SSM constant parameters ---
        # A is constrained negative; store as log(-A) for stability
        self.A_log  = nn.Parameter(torch.full((NHEADS,), 0.5))   # -exp(A_log) < 0
        self.D_vec  = nn.Parameter(torch.ones(NHEADS))
        self.xproj  = nn.Parameter(torch.randn(MIMO, NHEADS, HDIM) * 0.02)
        self.outproj = nn.Parameter(torch.randn(MIMO, NHEADS, HDIM) * 0.02)
        self.zproj  = nn.Parameter(torch.randn(MIMO, NHEADS, HDIM) * 0.02)

        # --- Per-step input projections ---
        # Projects D_MODEL → x(H*D) + z(H*D) + B(R*H*N) + C(R*H*N) + dt(H) + trap(H)
        d_xz  = 2 * NHEADS * HDIM               # 512
        d_bc  = 2 * MIMO * NHEADS * DSTATE      # 512
        d_dt  = NHEADS                           # 4
        d_trap = NHEADS                          # 4
        self.in_proj  = nn.Linear(D_MODEL, d_xz + d_bc + d_dt + d_trap, bias=False)

        # Optional: layer norm before LM head
        self.norm = nn.LayerNorm(D_MODEL)

        # dt bias (initialised to give dt in [0.001, 0.1])
        dt_min, dt_max = 0.001, 0.1
        _dt = torch.exp(
            torch.rand(NHEADS) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))

    # ------------------------------------------------------------------
    def _get_A(self) -> torch.Tensor:
        """Return A with shape (1, NHEADS) — always negative."""
        return -self.A_log.abs().unsqueeze(0)   # (1, H)

    # ------------------------------------------------------------------
    def _project_step(self, emb: torch.Tensor):
        """
        Given emb (1, D_MODEL), return the per-step SSM tensors.

        Returns
        -------
        x      : (1, H, D)
        z      : (1, H, D)
        B      : (1, R, H, N)
        C      : (1, R, H, N)
        dt     : (1, H)   — positive via softplus
        trap   : (1, H)   — in (0,1) via sigmoid
        """
        proj = self.in_proj(emb)   # (1, d_xz + d_bc + d_dt + d_trap)

        d_xz   = 2 * NHEADS * HDIM
        d_half = NHEADS * HDIM
        d_bc   = MIMO * NHEADS * DSTATE

        x_flat  = proj[:, :d_half]
        z_flat  = proj[:, d_half:d_xz]
        B_flat  = proj[:, d_xz          : d_xz + d_bc]
        C_flat  = proj[:, d_xz + d_bc   : d_xz + 2*d_bc]
        dt_raw  = proj[:, d_xz + 2*d_bc : d_xz + 2*d_bc + NHEADS]
        trap_r  = proj[:, d_xz + 2*d_bc + NHEADS:]

        x    = x_flat.view(1, NHEADS, HDIM)
        z    = z_flat.view(1, NHEADS, HDIM)
        B    = B_flat.view(1, MIMO, NHEADS, DSTATE)
        C    = C_flat.view(1, MIMO, NHEADS, DSTATE)
        dt   = F.softplus(dt_raw + self.dt_bias)          # (1, H)
        trap = torch.sigmoid(trap_r)                       # (1, H)

        return x, z, B, C, dt, trap

    # ------------------------------------------------------------------
    def init_state(self):
        """Return zero-initialised SSM states."""
        state   = torch.zeros(1, NHEADS, HDIM, DSTATE)  # (B,H,D,N)
        k_state = torch.zeros(1, MIMO, NHEADS, DSTATE)  # (B,R,H,N) — Bstate
        v_state = torch.zeros(1, NHEADS, HDIM)          # (B,H,D)   — Xstate
        return state, k_state, v_state

    # ------------------------------------------------------------------
    def ssm_step_cpu(self, state, k_state, v_state,
                     x, z, B, C, dt, trap):
        """
        Pure-PyTorch reference SSM step.

        Implements the same arithmetic as selective_state_update_fused_ref_v2
        from src/hls/mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py, but using
        the tensor layout that matches model_config.h:

            state   (B, H, D, N)
            k_state (B, R, H, N)  — "Bstate" / previous-B
            v_state (B, H, D)     — "Xstate" / previous-x

        Returns
        -------
        out       : (1, H, D)
        new_state : (1, H, D, N)
        new_k     : (1, R, H, N)
        new_v     : (1, H, D)
        """
        A      = self._get_A()                        # (1, H)
        D_vec  = self.D_vec                           # (H,)
        xproj  = self.xproj                           # (R, H, D)
        outproj = self.outproj                        # (R, H, D)
        zproj  = self.zproj                           # (R, H, D)

        # Bstate = k_state, Xstate = v_state
        Bstate = k_state   # (1, R, H, N)
        Xstate = v_state   # (1, H, D)

        alpha = torch.exp(A * dt)                     # (1, H)
        beta  = (1.0 - trap) * dt * alpha             # (1, H)
        gamma = trap * dt                             # (1, H)

        # x projected: (1, R, H, D) via xproj (R, H, D)
        x_proj  = x.unsqueeze(1) * xproj.unsqueeze(0)   # (1, R, H, D)
        xs_proj = Xstate.unsqueeze(1) * xproj.unsqueeze(0)  # (1, R, H, D)

        # Broadcast gamma/beta: (1,H) → (1,1,H,1) for (B,R,H,D) tensors
        g = gamma[:, None, :, None]   # (1, 1, H, 1)
        b = beta[:,  None, :, None]   # (1, 1, H, 1)

        # State update: (1, H, D, N)
        #   new_state = state * alpha + einsum(x_proj*gamma, B) + einsum(xs_proj*beta, Bstate)
        xBt  = torch.einsum('brhd,brhn->bhdn', x_proj  * g, B)
        xsBt = torch.einsum('brhd,brhn->bhdn', xs_proj * b, Bstate)

        alpha4d    = alpha.unsqueeze(-1).unsqueeze(-1)  # (1, H, 1, 1)
        new_state  = state * alpha4d + xBt + xsBt       # (1, H, D, N)

        # Output: C projection (1, R, H, D)  then sum ranks
        out_r = torch.einsum('bhdn,brhn->brhd', new_state, C)  # (1, R, H, D)

        # Skip connection: D * (x · xproj)
        out_r = out_r + D_vec[None, None, :, None] * x_proj  # (1, R, H, D)

        # SiLU gate
        z_proj = z.unsqueeze(1) * zproj.unsqueeze(0)   # (1, R, H, D)
        out_r  = out_r * z_proj * torch.sigmoid(z_proj) # (1, R, H, D)

        # Output projection: reduce R
        out = torch.einsum('brhd,rhd->bhd', out_r, outproj)  # (1, H, D)

        new_k = B.clone()   # next Bstate = this step's B
        new_v = x.clone()   # next Xstate = this step's x

        return out, new_state, new_k, new_v

    # ------------------------------------------------------------------
    def forward_token(self, token_id: int, state, k_state, v_state, use_fpga=False):
        """
        Process a single token through the model.

        Returns
        -------
        logits  : (vocab_size,)  — unnormalised log-probs for next token
        state   : updated SSM state   (1, H, D, N)
        k_state : updated Bstate      (1, R, H, N)
        v_state : updated Xstate      (1, H, D)
        """
        tok = torch.tensor([[token_id]], dtype=torch.long)
        emb = self.embedding(tok).squeeze(0)   # (1, D_MODEL)

        x, z, B, C, dt, trap = self._project_step(emb)

        if use_fpga:
            # Single-step via FPGA — package as a 1-step batch
            out_flat, new_state, new_k, new_v = fpga_run_nsteps(
                self, state, k_state, v_state,
                [(x, z, B, C, dt, trap)],
                nsteps=1,
            )
            out = out_flat[0]   # (1, H, D)
        else:
            out, new_state, new_k, new_v = self.ssm_step_cpu(
                state, k_state, v_state, x, z, B, C, dt, trap)

        # Flatten H,D → D_MODEL and apply output norm + LM head
        out_flat = out.view(1, D_MODEL)          # (1, D_MODEL)
        out_flat = self.norm(out_flat)
        logits   = self.lm_head(out_flat).squeeze(0)   # (vocab_size,)

        return logits, new_state, new_k, new_v


# ---------------------------------------------------------------------------
# FPGA communication helpers
# ---------------------------------------------------------------------------

def _write_exact(fd: int, data: bytes):
    total = 0
    mv = memoryview(data)
    while total < len(data):
        n = os.write(fd, mv[total:])
        if n <= 0:
            raise IOError("FPGA write returned 0 bytes")
        total += n


def _read_exact(fd: int, n_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = os.read(fd, n_bytes - len(buf))
        if not chunk:
            raise IOError("FPGA read returned 0 bytes")
        buf.extend(chunk)
    return bytes(buf)


def _pack(t: torch.Tensor) -> bytes:
    """Flatten a tensor to raw fp32 bytes (C-contiguous)."""
    return t.detach().cpu().float().contiguous().numpy().tobytes()


def _unpack(raw: bytes, count: int) -> torch.Tensor:
    """Unpack raw fp32 bytes into a 1-D float tensor."""
    return torch.frombuffer(raw, dtype=torch.float32).clone()[:count]


def fpga_run_nsteps(model: Mamba3LM, state, k_state, v_state,
                    per_step_inputs: list, nsteps: int, fdw: int, fdr: int):
    """
    Send `nsteps` worth of SSM inputs to the FPGA, collect outputs.

    Parameters
    ----------
    per_step_inputs : list of (x, z, B, C, dt, trap) tuples, length == nsteps

    Returns
    -------
    out_list  : list of (1, H, D) tensors, one per step
    new_state : (1, H, D, N)
    new_k     : (1, R, H, N)
    new_v     : (1, H, D)
    """
    A_const = model._get_A().expand(BATCH, NHEADS)  # (1, H)

    # --- Build the single send buffer ---
    parts = []

    # Constants (loaded once)
    parts.append(_pack(A_const))                      # A     (B, H)
    parts.append(_pack(model.D_vec))                  # D     (H,)
    parts.append(_pack(model.xproj))                  # xproj (R,H,D)
    parts.append(_pack(model.outproj))                # outproj
    parts.append(_pack(model.zproj))                  # zproj

    # Initial state
    parts.append(_pack(state))                        # (1,H,D,N)

    # Per-step tensors
    prev_k = k_state
    prev_v = v_state
    for x, z, B, C, dt, trap in per_step_inputs:
        parts.append(_pack(prev_k))                   # Bstate (1,R,H,N)
        parts.append(_pack(prev_v))                   # Xstate (1,H,D)
        parts.append(_pack(B))                        # B      (1,R,H,N)
        parts.append(_pack(C))                        # C      (1,R,H,N)
        parts.append(_pack(x))                        # x      (1,H,D)
        parts.append(_pack(dt))                       # dt     (1,H)
        parts.append(_pack(trap))                     # trap   (1,H)
        parts.append(_pack(z))                        # z      (1,H,D)
        prev_k = B.clone()
        prev_v = x.clone()

    send_buf = b"".join(parts)

    total_out_floats = nsteps * X_SIZE + STATE_SIZE
    total_out_bytes  = total_out_floats * 4

    try:
        recv_holder = [None]

        def _writer():
            _write_exact(fdw, send_buf)

        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()
        recv_holder[0] = _read_exact(fdr, total_out_bytes)
        writer_thread.join()
    except Exception as e:
        raise e

    # --- Unpack outputs ---
    raw = recv_holder[0]
    offset = 0
    out_list = []
    for _ in range(nsteps):
        floats = _unpack(raw[offset: offset + X_SIZE * 4], X_SIZE)
        out_list.append(floats.view(1, NHEADS, HDIM))
        offset += X_SIZE * 4

    state_raw  = _unpack(raw[offset: offset + STATE_SIZE * 4], STATE_SIZE)
    new_state  = state_raw.view(1, NHEADS, HDIM, DSTATE)

    # Maintain k/v from last step
    last_x, _, last_B, _, _, _ = per_step_inputs[-1]
    new_k = last_B.clone()
    new_v = last_x.clone()

    return out_list, new_state, new_k, new_v


def fpga_available() -> bool:
    return os.path.exists(FPGA_WRITE_DEV) and os.path.exists(FPGA_READ_DEV)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def process_prompt(model: Mamba3LM, prompt_ids: list[int], use_fpga: bool):
    """
    Run prompt tokens through the model to build up SSM state.
    Returns (state, k_state, v_state, last_logits).
    """
    state, k_state, v_state = model.init_state()
    logits = torch.zeros(model.vocab_size)
    for tid in prompt_ids:
        logits, state, k_state, v_state = model.forward_token(
            tid, state, k_state, v_state, use_fpga=False)  # always CPU for prompt
    return state, k_state, v_state, logits


def generate(model: Mamba3LM, prompt_ids: list[int],
             max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = 1.0, top_k: int = 50,
             use_fpga: bool = True, fdw: int = -1, fdr: int = -1) -> list[int]:
    """
    Autoregressive generation with FPGA-accelerated SSM steps (64 at a time).

    Returns the list of newly generated token IDs.
    """
    model.eval()
    with torch.no_grad():

        # --- Warm up state on the prompt ---
        print("Processing prompt...", flush=True)
        state, k_state, v_state, last_logits = process_prompt(
            model, prompt_ids, use_fpga=False)

        generated = []
        # Seed with the last prompt token (re-used as the first generation input)
        next_tok = prompt_ids[-1] if prompt_ids else 0

        steps_remaining = max_new_tokens
        step_global = 0

        while steps_remaining > 0:
            batch = min(NSTEPS, steps_remaining)

            if use_fpga:
                # -----------------------------------------------------------
                # FPGA path: collect `batch` per-step input tuples,
                # then fire one FPGA invocation.
                # We need to pad to NSTEPS if batch < NSTEPS (FPGA DUT
                # always runs exactly NSTEPS steps due to compile-time const).
                # We pad with the last real step repeated.
                # -----------------------------------------------------------
                per_step = []
                sampled_toks = []
                cur_tok = next_tok
                cur_state, cur_k, cur_v = state, k_state, v_state

                for s in range(NSTEPS):
                    tok = torch.tensor([[cur_tok]], dtype=torch.long)
                    emb = model.embedding(tok).squeeze(0)
                    x, z, B, C, dt, trap = model._project_step(emb)
                    per_step.append((x, z, B, C, dt, trap))

                    # We need logits to pick the next token even mid-batch.
                    # Use a CPU step for each token to get logits; the FPGA
                    # call will update the state at end of the batch.
                    out_cpu, cur_state_next, _, _ = model.ssm_step_cpu(
                        cur_state, cur_k, cur_v, x, z, B, C, dt, trap)
                    # Advance CPU shadow state so subsequent CPU logits are consistent.
                    # The authoritative state update for the batch will come from the FPGA.
                    cur_state = cur_state_next
                    cur_k = B.clone()
                    cur_v = x.clone()
                    out_flat = out_cpu.view(1, D_MODEL)
                    logits_s = model.lm_head(model.norm(out_flat)).squeeze(0)

                    # Sample next token
                    if temperature != 1.0:
                        logits_s = logits_s / temperature
                    if top_k > 0:
                        kth = torch.topk(logits_s, min(top_k, logits_s.size(-1))).values[-1]
                        logits_s = logits_s.masked_fill(logits_s < kth, float('-inf'))
                    probs   = F.softmax(logits_s, dim=-1)
                    cur_tok = torch.multinomial(probs, 1).item()

                    if s < batch:
                        sampled_toks.append(cur_tok)

                # Fire FPGA to get the authoritative new state
                try:
                    _, new_state, new_k, new_v = fpga_run_nsteps(
                        model, state, k_state, v_state, per_step, nsteps=NSTEPS, fdw=fdw, fdr=fdr)
                    state, k_state, v_state = new_state, new_k, new_v
                except Exception as exc:
                    print(f"\n[FPGA error: {exc}; falling back to CPU state]",
                          flush=True)
                    use_fpga = False
                    state, k_state, v_state = cur_state, cur_k, cur_v   # already updated token-by-token above

                generated.extend(sampled_toks[:batch])
                next_tok = sampled_toks[batch - 1] if sampled_toks else cur_tok

            else:
                # -----------------------------------------------------------
                # CPU fallback: one token at a time
                # -----------------------------------------------------------
                tok = torch.tensor([[next_tok]], dtype=torch.long)
                emb = model.embedding(tok).squeeze(0)
                x, z, B, C, dt, trap = model._project_step(emb)
                out, state, k_state, v_state = model.ssm_step_cpu(
                    state, k_state, v_state, x, z, B, C, dt, trap)
                out_flat = out.view(1, D_MODEL)
                logits   = model.lm_head(model.norm(out_flat)).squeeze(0)

                if temperature != 1.0:
                    logits = logits / temperature
                if top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                    logits = logits.masked_fill(logits < kth, float('-inf'))
                probs    = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
                generated.append(next_tok)
                batch = 1

            step_global += batch
            steps_remaining -= batch

            # Stream output to terminal as bytes
            chunk = decode(generated[-batch:])
            print(chunk, end="", flush=True)

        print()  # newline after generation
        return generated


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_or_init_model() -> Mamba3LM:
    model = Mamba3LM(vocab_size=VOCAB)
    weights_path = os.path.abspath(WEIGHTS_PATH)

    if os.path.isfile(weights_path):
        print(f"Loading weights from {weights_path} ...", flush=True)
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Support both raw state-dict and {'model': state_dict} checkpoints
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            model.load_state_dict(ckpt, strict=False)
            print("Weights loaded.", flush=True)
        except Exception as exc:
            print(f"[Warning] Could not load weights ({exc}). Using random weights.",
                  flush=True)
    else:
        print(f"No weights file found at {weights_path}. Using random weights for testing.",
              flush=True)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Mamba-3 FPGA Text Generator")
    print(f"  Config: H={NHEADS} D={HDIM} N={DSTATE} R={MIMO} NSTEPS={NSTEPS}")
    print("=" * 60)

    model = load_or_init_model()

    on_fpga = fpga_available()
    fdw, fdr = -1, -1
    if on_fpga:
        print(f"FPGA device detected ({FPGA_WRITE_DEV}). Using FPGA acceleration.")
        fdr = os.open(FPGA_READ_DEV,  os.O_RDONLY)
        fdw = os.open(FPGA_WRITE_DEV, os.O_WRONLY)
    else:
        print("FPGA device not found. Running in CPU-only mode.")

    print()

    try:
        while True:
            try:
                prompt_text = input("Enter prompt (Ctrl-C or Ctrl-D to quit): ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not prompt_text:
                print("[Empty prompt, please try again]")
                continue

            prompt_ids = encode(prompt_text)
            print("\n--- Generated output ---")
            print(prompt_text, end="", flush=True)

            try:
                generate(
                    model,
                    prompt_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.8,
                    top_k=50,
                    use_fpga=on_fpga,
                    fdw=fdw,
                    fdr=fdr,
                )
            except KeyboardInterrupt:
                print("\n[Generation interrupted]")

            print("--- End of output ---\n")
    finally:
        if fdw != -1: os.close(fdw)
        if fdr != -1: os.close(fdr)

if __name__ == "__main__":
    main()
