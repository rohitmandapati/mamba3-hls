#==========================================================================
# Makefile
#==========================================================================
# @brief: Compiles and synthesizes the Mamba-3 step function accelerator.
#
# Usage:
#   make csim                      — baseline fp32 software-side C simulation
#   make dut                       — baseline HLS synthesis to RTL
#   make bitstream                 — baseline bitstream build
#   make csim MAMBA3_ENABLE_HLS_PRAGMAS=1
#   make dut  MAMBA3_ENABLE_HLS_PRAGMAS=1
#   make csim MAMBA3_USE_FIXED_POINT=1
#   make dut  MAMBA3_USE_FIXED_POINT=1 MAMBA3_ENABLE_HLS_PRAGMAS=1
#   make show-config               — print the active build settings
#   make clean                     — remove all build artifacts

# Define paths and sources, absolute paths required
ROOT_DIR=$(shell pwd)
SRC_PATH=$(shell pwd)/src/hls
BUILD_PATH=$(shell pwd)/build
HW_DUT_PATH=$(BUILD_PATH)/hw_dut
WORKSPACE_PATH=$(shell pwd)/workspace
DUT_SRC=$(SRC_PATH)/mamba3_step.cpp
TEST_SRC=$(SRC_PATH)/mamba3_step_test.cpp
KRIA_DIR=$(shell pwd)/kria_dir
KRIA_SRC=$(shell pwd)/src/kria
RESULTS_DIR=$(shell pwd)/result
CONFIG_STAMP=$(BUILD_PATH)/.mamba3_config

# Extract Vivado HLS include path
XILINX_VITIS?=/tools/Xilinx/2025.1/Vitis
XILINX_VITIS_HLS?=/tools/Xilinx/2025.1/Vitis_HLS
XILINX_VIVADO?=/tools/Xilinx/2025.1/Vivado
XIL_HLS=source $(XILINX_VITIS)/settings64.sh; vitis-run --mode hls --tcl 	
XIL_VIVADO=source $(XILINX_VIVADO)/settings64.sh; vivado -mode batch -source
VHLS_INC=$(if $(wildcard $(XILINX_VITIS)/include),-I$(XILINX_VITIS)/include) $(if $(wildcard $(XILINX_VITIS_HLS)/include),-I$(XILINX_VITIS_HLS)/include)

CFLAGS = -g ${VHLS_INC} -I$(SRC_PATH) -DHLS_NO_XIL_FPO_LIB -std=c++11 -O3

TCL_NAME?=build_dut.tcl

MAMBA3_ENABLE_HLS_PRAGMAS ?= 0
MAMBA3_PIPELINE_II ?= 1
MAMBA3_UNROLL_MIMO ?= 1
MAMBA3_USE_FIXED_POINT ?= 0
MAMBA3_FIXED_TOTAL_BITS ?= 24
MAMBA3_FIXED_INT_BITS ?= 10
MAMBA3_USE_PWL_NONLINEARITIES ?= $(MAMBA3_USE_FIXED_POINT)
MAMBA3_USE_CORDIC ?= 0
MAMBA3_NSTEPS ?= 1
EXTRA_MAMBA3_DEFINES ?=

MAMBA3_DEFINES = \
	-DMAMBA3_ENABLE_HLS_PRAGMAS=$(MAMBA3_ENABLE_HLS_PRAGMAS) \
	-DMAMBA3_PIPELINE_II=$(MAMBA3_PIPELINE_II) \
	-DMAMBA3_UNROLL_MIMO=$(MAMBA3_UNROLL_MIMO) \
	-DMAMBA3_USE_FIXED_POINT=$(MAMBA3_USE_FIXED_POINT) \
	-DMAMBA3_FIXED_TOTAL_BITS=$(MAMBA3_FIXED_TOTAL_BITS) \
	-DMAMBA3_FIXED_INT_BITS=$(MAMBA3_FIXED_INT_BITS) \
	-DMAMBA3_USE_PWL_NONLINEARITIES=$(MAMBA3_USE_PWL_NONLINEARITIES) \
	-DMAMBA3_USE_CORDIC=$(MAMBA3_USE_CORDIC) \
	-DMAMBA3_NSTEPS=$(MAMBA3_NSTEPS) \
	-DMAMBA3_FORCE_XILINX_HEADERS=1 \
	$(EXTRA_MAMBA3_DEFINES)

CSIM_CFLAGS = $(CFLAGS) $(MAMBA3_DEFINES)
HLS_FILE_CFLAGS = -std=c++11 $(MAMBA3_DEFINES)

.PHONY: all help show-config clean FORCE

help:
	@echo "Targets:"
	@echo "  make csim"
	@echo "  make dut"
	@echo "  make bitstream"
	@echo "  make show-config"
	@echo ""
	@echo "Common overrides:"
	@echo "  MAMBA3_ENABLE_HLS_PRAGMAS=1"
	@echo "  MAMBA3_USE_FIXED_POINT=1"
	@echo "  MAMBA3_FIXED_TOTAL_BITS=24"
	@echo "  MAMBA3_FIXED_INT_BITS=8"
	@echo "  MAMBA3_PIPELINE_II=1"
	@echo "  MAMBA3_UNROLL_MIMO=1"
	@echo "  MAMBA3_USE_CORDIC=1"
	@echo "  MAMBA3_NSTEPS=4"
	@echo ""
	@echo "Examples:"
	@echo "  make csim MAMBA3_ENABLE_HLS_PRAGMAS=1"
	@echo "  make dut MAMBA3_ENABLE_HLS_PRAGMAS=1"
	@echo "  make csim MAMBA3_USE_FIXED_POINT=1"
	@echo "  make csim MAMBA3_USE_CORDIC=1"
	@echo "  make csim MAMBA3_NSTEPS=4"
	@echo "  make bitstream MAMBA3_ENABLE_HLS_PRAGMAS=1 MAMBA3_USE_FIXED_POINT=1"

show-config:
	@echo "MAMBA3_ENABLE_HLS_PRAGMAS=$(MAMBA3_ENABLE_HLS_PRAGMAS)"
	@echo "MAMBA3_PIPELINE_II=$(MAMBA3_PIPELINE_II)"
	@echo "MAMBA3_UNROLL_MIMO=$(MAMBA3_UNROLL_MIMO)"
	@echo "MAMBA3_USE_FIXED_POINT=$(MAMBA3_USE_FIXED_POINT)"
	@echo "MAMBA3_FIXED_TOTAL_BITS=$(MAMBA3_FIXED_TOTAL_BITS)"
	@echo "MAMBA3_FIXED_INT_BITS=$(MAMBA3_FIXED_INT_BITS)"
	@echo "MAMBA3_USE_PWL_NONLINEARITIES=$(MAMBA3_USE_PWL_NONLINEARITIES)"
	@echo "MAMBA3_USE_CORDIC=$(MAMBA3_USE_CORDIC)"
	@echo "MAMBA3_NSTEPS=$(MAMBA3_NSTEPS)"
	@echo "EXTRA_MAMBA3_DEFINES=$(EXTRA_MAMBA3_DEFINES)"

all: bitstream deploy

csim: ${RESULTS_DIR}/mamba3_step_csim.txt

dut: ${HW_DUT_PATH}/dut.v

bitstream: ${BUILD_PATH}/bitstream.bit deploy

deploy: ${BUILD_PATH}/bitstream.bit
	mkdir -p ${KRIA_DIR}
	cp ${BUILD_PATH}/bitstream.bit ${KRIA_DIR}/
	cp -r ${SRC_PATH}/* ${KRIA_DIR}/
	cp -r ${KRIA_SRC}/* ${KRIA_DIR}/

clean:
	rm -rf ${BUILD_PATH}
	rm -rf *.dat *.prj *.log *.jou logs/ .Xil/ ${KRIA_DIR}/ ${RESULTS_DIR}

# === Internal targets ===

FORCE:

$(CONFIG_STAMP): FORCE
	mkdir -p ${BUILD_PATH}
	@tmp_file="$@.tmp"; \
	printf '%s\n' 'CSIM_CFLAGS=$(CSIM_CFLAGS)' 'HLS_FILE_CFLAGS=$(HLS_FILE_CFLAGS)' > "$$tmp_file"; \
	if ! test -f "$@" || ! cmp -s "$$tmp_file" "$@"; then \
		mv "$$tmp_file" "$@"; \
	else \
		rm -f "$$tmp_file"; \
	fi

${BUILD_PATH}/mamba3_step: $(CONFIG_STAMP) $(DUT_SRC) $(TEST_SRC)
	mkdir -p ${BUILD_PATH}
	g++ ${CSIM_CFLAGS} $(DUT_SRC) $(TEST_SRC) -o $@ -lrt -lm

${RESULTS_DIR}/mamba3_step_csim.txt: ${BUILD_PATH}/mamba3_step
	@echo "Running Mamba-3 step csim..."
	mkdir -p ${RESULTS_DIR}
	$< | tee $@

${HW_DUT_PATH}/dut.v: $(CONFIG_STAMP) $(DUT_SRC)
	@echo "================================================================="
	@echo "Synthesizing Mamba-3 step with $(TCL_NAME)..."
	@echo "================================================================="
	mkdir -p ${HW_DUT_PATH}
	SRC_PATH=${SRC_PATH} TEST_PATH=${SRC_PATH} SKIP_CSIM=1 SKIP_COSIM=1 HLS_FILE_CFLAGS='$(HLS_FILE_CFLAGS)' $(XIL_HLS) artifacts/${TCL_NAME}
	cp mamba3_step.prj/solution1/impl/ip/hdl/verilog/* ${HW_DUT_PATH}/

${BUILD_PATH}/bitstream.bit: ${HW_DUT_PATH}/dut.v
	@echo "================================================================="
	@echo "Creating bitstream with Vivado..."
	@echo "================================================================="
	${XIL_VIVADO} artifacts/build_bitstream.tcl
	cp kria.prj/kria.prj.runs/impl_1/design_1_wrapper.bit $@
