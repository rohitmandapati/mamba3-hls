#*********************************************************************************************
# build_dut.tcl 
# @brief: Tcl script for synthesizing the Mamba-3 step function accelerator.
#*********************************************************************************************

# Get test path from environment variable TEST_PATH, default to "./test/" directory if not set

set test_path "./test/"
if {[info exists ::env(TEST_PATH)]} {
    set test_path $::env(TEST_PATH)
}

# Get src path from environment variable SRC_PATH, default to "./src/" directory if not set
set src_path "./src/"
if {[info exists ::env(SRC_PATH)]} {
    set src_path $::env(SRC_PATH)
}

# Get project parent path from environment variable PROJ_PATH, default to current directory if not set
set proj_path "."
if {[info exists ::env(PROJ_PATH)]} {
    set proj_path $::env(PROJ_PATH)
}

# Verify is environment variable SKIP_CSIM is set to "1" or not
set skip_csim "0"
if {[info exists ::env(SKIP_CSIM)]} {
    set skip_csim $::env(SKIP_CSIM)
}

# Verify is environment variable SKIP_COSIM is set to "1" or not
set skip_cosim "0"
if {[info exists ::env(SKIP_COSIM)]} {
    set skip_cosim $::env(SKIP_COSIM)
}

# Extra compile flags for design/tb files, passed from the top-level Makefile.
set file_cflags "-std=c++11"
if {[info exists ::env(HLS_FILE_CFLAGS)]} {
    set file_cflags $::env(HLS_FILE_CFLAGS)
}

# Open/reset the project — catch errors from NFS stale .nfs* ghost files
# that can't be deleted on shared filesystems.
if {[catch {open_project -reset "$proj_path/mamba3_step.prj"} err]} {
    puts "WARNING: open_project -reset had an error (likely NFS ghost file): $err"
    puts "Retrying without -reset..."
    catch {file delete -force "$proj_path/mamba3_step.prj"}
    open_project "$proj_path/mamba3_step.prj"
}

# Top function of the design is "dut"
set_top dut

# Add design and testbench files
add_files -cflags $file_cflags $src_path/mamba3_step.cpp
add_files -cflags $file_cflags -tb $test_path/mamba3_step_test.cpp

open_solution "solution1"

# use KV260 device (Kria K26 SOM - Zynq UltraScale+ MPSoC)
set_part xck26-sfvc784-2LV-c

# Target clock period is 10ns
create_clock -period 10 -name default

### You can insert your own directives here ###

############################################

# Simulate the C++ design
if {$skip_csim == "1"} {
    # Skip C simulation if SKIP_CSIM is set to "1"
    puts "Skipping C simulation as SKIP_CSIM is set to 1"
} else {
    # C simulation
    csim_design -O
}

# Synthesize the design
csynth_design

if {$skip_cosim == "1"} {
    # Skip C-RTL cosimulation if SKIP_COSIM is set to "1"
    puts "Skipping C-RTL cosimulation as SKIP_COSIM is set to 1"
} else {
    # C-RTL cosimulation
    cosim_design
}

export_design -format ip_catalog

exit
