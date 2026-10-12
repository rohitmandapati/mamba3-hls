#==========================================================================
# Makefile
#==========================================================================
# @brief: Build/run helpers for the Kria host application.

CXX := g++-13
HLS_INC := .
MAMBA3_NSTEPS ?= 64
CXXFLAGS := -I$(HLS_INC) -O3 -std=c++11 -DMAMBA3_NSTEPS=$(MAMBA3_NSTEPS)
LDFLAGS  := -lpthread
HOST_BIN := mamba3-host

.PHONY: all host fpga clean

all: host

$(HOST_BIN): host.cpp
	@echo "Compiling Mamba-3 host program"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

host: $(HOST_BIN)

fpga: $(HOST_BIN)
	@echo "Running Mamba-3 host program"
	./$<

clean:
	rm -f $(HOST_BIN)
