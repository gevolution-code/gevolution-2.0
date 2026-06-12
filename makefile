# Portable GNU Make entry point for gevolution 2.0.
#
# Configuration precedence:
#   1. config/defaults.mk
#   2. CONFIG (config/local.mk by default, if present)
#   3. command-line overrides

.DEFAULT_GOAL := gevolution

CONFIG ?= config/local.mk

include config/defaults.mk
-include $(CONFIG)

EXEC := gevolution
SOURCE := main.cu
HEADERS := $(wildcard *.hpp)
BUILD_CONFIG_FILES := makefile config/defaults.mk $(wildcard $(CONFIG))

LATFIELD2_DEFINES := FFT3D HDF5
GEVOLUTION_DEFINES := $(COMMON_DEFINES) $(EXTRA_DEFINES)

ifeq ($(PARALLEL_HDF5),1)
LATFIELD2_DEFINES += H5_HAVE_PARALLEL
HDF5_PKG := $(PARALLEL_HDF5_PKG)
else
HDF5_PKG := $(SERIAL_HDF5_PKG)
endif

ifeq ($(PRECISION),single)
LATFIELD2_DEFINES += SINGLE
FFTW_PKG := fftw3f
FFTW_LIB := fftw3f
else ifeq ($(PRECISION),double)
FFTW_PKG := fftw3
FFTW_LIB := fftw3
else
$(error PRECISION must be 'single' or 'double', got '$(PRECISION)')
endif

ifeq ($(ENABLE_CLASS),1)
GEVOLUTION_DEFINES += HAVE_CLASS
FEATURE_LDLIBS += -lclass
endif

ifeq ($(ENABLE_HEALPIX),1)
GEVOLUTION_DEFINES += HAVE_HEALPIX
HEALPIX_PKG := chealpix
ifeq ($(USE_PKG_CONFIG),0)
FEATURE_LDLIBS += -lchealpix
endif
endif

PKG_CONFIG_PACKAGES := $(HDF5_PKG) gsl $(FFTW_PKG) $(HEALPIX_PKG)
ifeq ($(USE_PKG_CONFIG),1)
PKG_CPPFLAGS := $(shell $(PKG_CONFIG) --cflags $(PKG_CONFIG_PACKAGES) 2>/dev/null)
PKG_LDLIBS := $(shell $(PKG_CONFIG) --libs $(PKG_CONFIG_PACKAGES) 2>/dev/null)
LCMAP_PKG_CPPFLAGS := $(shell $(PKG_CONFIG) --cflags chealpix healpix_cxx cfitsio gsl 2>/dev/null)
LCMAP_PKG_LDLIBS := $(shell $(PKG_CONFIG) --libs chealpix healpix_cxx cfitsio gsl 2>/dev/null)
else
CORE_LDLIBS := -lhdf5 -lgsl -lgslcblas -lm -l$(FFTW_LIB)
LCMAP_CORE_LDLIBS := -lhealpix_cxx -lchealpix -lcfitsio -lgsl -lgslcblas -lm
endif

DEFINE_FLAGS := $(addprefix -D,$(LATFIELD2_DEFINES) $(GEVOLUTION_DEFINES))
BUILD_CPPFLAGS := -I. -I$(LATFIELD2_DIR) $(PKG_CPPFLAGS) $(CPPFLAGS) $(DEFINE_FLAGS)
BUILD_NVCCFLAGS := -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(MPICXX) --extended-lambda -Xcompiler -fopenmp $(NVCCFLAGS)
BUILD_CXXFLAGS := -std=c++17 -fopenmp $(CXXFLAGS)
BUILD_LDLIBS := $(PKG_LDLIBS) $(CORE_LDLIBS) $(FEATURE_LDLIBS) -lcufft $(LDLIBS)
BUILD_UTILITY_CPPFLAGS := -I. $(PKG_CPPFLAGS) $(CPPFLAGS) $(addprefix -D,$(GEVOLUTION_DEFINES))
BUILD_LCMAP_CPPFLAGS := $(BUILD_UTILITY_CPPFLAGS) $(LCMAP_PKG_CPPFLAGS) $(LCMAP_CPPFLAGS)
BUILD_LCMAP_LDLIBS := $(LCMAP_PKG_LDLIBS) $(LCMAP_CORE_LDLIBS) $(LCMAP_LDLIBS)

.PHONY: help print-config check-config parser-tests clean

help:
	@printf '%s\n' \
	  'gevolution 2.0 build targets:' \
	  '  gevolution      Build the GPU simulation executable (default)' \
	  '  unit-tests      Build GPU unit tests' \
	  '  parser-tests    Build and run parser tests; no CUDA configuration required' \
	  '  lccat           Build the particle light-cone catalogue utility' \
	  '  lcmap           Build the HEALPix/FITS map utility' \
	  '  print-config    Show the resolved build configuration' \
	  '  check-config    Validate configuration for GPU targets' \
	  '  clean           Remove built executables' \
	  '' \
	  'Configuration examples:' \
	  '  cp config/local.mk.example config/local.mk' \
	  '  make print-config' \
	  '  make CONFIG=config/alps.mk' \
	  '  make CUDA_ARCH=sm_86 EXTRA_DEFINES="FIXED_ICS VELOCITY"'

print-config:
	@printf '%-22s %s\n' \
	  'CONFIG' '$(CONFIG)' \
	  'NVCC' '$(NVCC)' \
	  'CXX' '$(CXX)' \
	  'MPICXX' '$(MPICXX)' \
	  'CUDA_ARCH' '$(CUDA_ARCH)' \
	  'LATFIELD2_DIR' '$(LATFIELD2_DIR)' \
	  'PRECISION' '$(PRECISION)' \
	  'PARALLEL_HDF5' '$(PARALLEL_HDF5)' \
	  'ENABLE_CLASS' '$(ENABLE_CLASS)' \
	  'ENABLE_HEALPIX' '$(ENABLE_HEALPIX)' \
	  'USE_PKG_CONFIG' '$(USE_PKG_CONFIG)' \
	  'PKG_CONFIG_PACKAGES' '$(PKG_CONFIG_PACKAGES)' \
	  'DEFINES' '$(LATFIELD2_DEFINES) $(GEVOLUTION_DEFINES)' \
	  'CPPFLAGS' '$(BUILD_CPPFLAGS)' \
	  'NVCCFLAGS' '$(BUILD_NVCCFLAGS)' \
	  'LDFLAGS' '$(LDFLAGS)' \
	  'LDLIBS' '$(BUILD_LDLIBS)'

check-config:
	@test -n "$(strip $(CUDA_ARCH))" || { \
	  echo "error: CUDA_ARCH is required for GPU targets."; \
	  echo "Set it in config/local.mk, select a profile, or run make CUDA_ARCH=sm_XX."; \
	  exit 2; \
	}
	@test -f "$(LATFIELD2_DIR)/LATfield2.hpp" || { \
	  echo "error: LATfield2.hpp not found under LATFIELD2_DIR=$(LATFIELD2_DIR)"; \
	  exit 2; \
	}
ifeq ($(USE_PKG_CONFIG),1)
	@$(PKG_CONFIG) --exists $(PKG_CONFIG_PACKAGES) || { \
	  echo "error: pkg-config could not resolve: $(PKG_CONFIG_PACKAGES)"; \
	  exit 2; \
	}
endif

$(EXEC): $(SOURCE) $(HEADERS) $(BUILD_CONFIG_FILES) | check-config
	$(NVCC) $(BUILD_CPPFLAGS) $(BUILD_NVCCFLAGS) $< -o $@ $(LDFLAGS) $(BUILD_LDLIBS)

unit-tests: unit_tests.cu $(HEADERS) $(BUILD_CONFIG_FILES) | check-config
	$(NVCC) $(BUILD_CPPFLAGS) $(BUILD_NVCCFLAGS) $< -o $@ $(LDFLAGS) $(BUILD_LDLIBS) -DGADGET_LENGTH_CONVERSION=1 -DGADGET_VELOCITY_CONVERSION=1

parser-tests: tests/parser_tests.cpp parser.hpp metadata.hpp
	$(CXX) -std=c++17 -Wall -Wextra -pedantic $< -o /tmp/gevolution-parser-tests
	/tmp/gevolution-parser-tests

lccat: lccat.cpp metadata.hpp parser.hpp $(BUILD_CONFIG_FILES)
	$(CXX) $(BUILD_UTILITY_CPPFLAGS) $(BUILD_CXXFLAGS) $< -o $@ $(LDFLAGS)

lcmap: lcmap.cpp metadata.hpp parser.hpp background.hpp $(BUILD_CONFIG_FILES)
	@if [ "$(USE_PKG_CONFIG)" = "1" ]; then \
	  $(PKG_CONFIG) --exists chealpix healpix_cxx cfitsio gsl || { \
	    echo "error: pkg-config could not resolve lcmap dependencies: chealpix healpix_cxx cfitsio gsl"; \
	    exit 2; \
	  }; \
	fi
	$(CXX) $(BUILD_LCMAP_CPPFLAGS) $(BUILD_CXXFLAGS) $< -o $@ $(LDFLAGS) $(BUILD_LCMAP_LDLIBS)

clean:
	-rm -f $(EXEC) unit-tests lccat lcmap

ifeq ($(ENABLE_ALPS_TARGETS),1)
include make/alps-targets.mk
endif
