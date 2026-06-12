# CSCS Alps / Grace Hopper profile.
# Load the matching CSCS programming environment before building.

CUDA_ARCH := sm_90
LATFIELD2_DIR := ../LATfield2

USE_PKG_CONFIG := 0
ENABLE_CLASS := 1
ENABLE_HEALPIX := 1
ENABLE_ALPS_TARGETS := 1

CPPFLAGS += \
	-I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/hdf5-1.14.5-iyjsbrml3dbr3l7cp65dgeclqlyfcdnn/include \
	-I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/gsl-2.8-pjzdxlsptkmjuvnrxif5x7ellp7rab3c/include \
	-I/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/fftw-3.3.10-3yw4wbosrsa2257uitrgpge6a3mfw7ck/include \
	-I/users/adamek/local_arm/include \
	-I../class_public/include \
	-I../class_public/external/HyRec2020 \
	-I../class_public/external/RecfastCLASS \
	-I../class_public/external/heating

LDFLAGS += \
	-L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/hdf5-1.14.5-iyjsbrml3dbr3l7cp65dgeclqlyfcdnn/lib \
	-L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/gsl-2.8-pjzdxlsptkmjuvnrxif5x7ellp7rab3c/lib \
	-L/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/fftw-3.3.10-3yw4wbosrsa2257uitrgpge6a3mfw7ck/lib \
	-L/users/adamek/local_arm/lib

LCMAP_CPPFLAGS += -I/users/adamek/local_arm/include/healpix_cxx

# Site and allocation-specific execution defaults used by make/alps-targets.mk.
ALPS_ACCOUNT ?= sm97
ALPS_PROFILE_ACCOUNT ?= go25
ALPS_PARTITION ?= debug
ALPS_SCRATCH ?= /capstor/scratch/cscs/adamek/testing
ALPS_THREADS ?= 72
