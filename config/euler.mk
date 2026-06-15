# ETH Zurich Euler / NVIDIA A100 profile.
# Load the required Euler modules before building. Dependency roots can be
# overridden on the make command line or inherited from the module environment.

CUDA_ARCH := sm_80

CUDA_HOME ?= $(CUDA_PATH)
HDF5_HOME ?= $(HDF5_DIR)
GSL_HOME ?= $(GSL_DIR)
FFTW_HOME ?= $(FFTW_DIR)
HEALPIX_HOME ?= ../../Healpix_3.83
CFITSIO_HOME ?= $(CFITSIO_DIR)
CLASS_HOME ?= ../class_public
LATFIELD2_HOME ?= ../LATfield2
LOCAL_HOME ?= $(HOME)/local

LATFIELD2_DIR := $(LATFIELD2_HOME)

PRECISION := single
PARALLEL_HDF5 := 1
USE_PKG_CONFIG := 0
ENABLE_CLASS := 1
ENABLE_HEALPIX := 1
ENABLE_ALPS_TARGETS := 0

CPPFLAGS += \
	-I$(HDF5_HOME)/include \
	-I$(GSL_HOME)/include \
	-I$(FFTW_HOME)/include \
	-I$(LOCAL_HOME)/include \
	-I$(CLASS_HOME)/include \
	-I$(CLASS_HOME)/external/HyRec2020 \
	-I$(CLASS_HOME)/external/RecfastCLASS \
	-I$(CLASS_HOME)/external/heating \
	-I$(CUDA_HOME)/include \
	-I$(HEALPIX_HOME)/include

LDFLAGS += \
	-L$(HDF5_HOME)/lib \
	-L$(GSL_HOME)/lib \
	-L$(FFTW_HOME)/lib \
	-L$(CLASS_HOME) \
	-L$(LOCAL_HOME)/lib \
	-L$(CUDA_HOME)/lib64 \
	-L$(HEALPIX_HOME)/lib \
	-L$(CFITSIO_HOME)/lib

LCMAP_CPPFLAGS += -I$(HEALPIX_HOME)/include/healpix_cxx
