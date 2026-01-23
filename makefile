# programming environment
COMPILER     := nvcc
INCLUDE      := -I../LATfield2 -I../class/include
INCLUDE      += $(addprefix -I,$(shell find ../class/external -type d))
LIB          := -lfftw3f -lm -lhdf5 -lgsl -lgslcblas -lcufft
LIB      	 += -L../class -lclass
HPXCXXLIB    := 

# target and source
EXEC         := gevolution
SOURCE       := main.cu
HEADERS      := $(wildcard *.hpp)

# mandatory compiler settings (LATfield2)
DLATFIELD2   := -DFFT3D -DHDF5

# optional compiler settings (LATfield2)
DLATFIELD2   += -DH5_HAVE_PARALLEL
#DLATFIELD2   += -DEXTERNAL_IO # enables I/O server (use with care)
DLATFIELD2   += -DSINGLE      # switches to single precision, use LIB -lfftw3f

# optional compiler settings (gevolution)
DGEVOLUTION  := -DPHINONLINEAR
DGEVOLUTION  += -DBENCHMARK
DGEVOLUTION  += -DEXACT_OUTPUT_REDSHIFTS
#DGEVOLUTION  += -DVELOCITY      # enables velocity field utilities
DGEVOLUTION  += -DCOLORTERMINAL
#DGEVOLUTION  += -DCHECK_B
DGEVOLUTION  += -DHAVE_CLASS    # requires LIB -lclass
#DGEVOLUTION  += -DHAVE_HEALPIX  # requires LIB -lchealpix
DGEVOLUTION  += -DGRADIENT_ORDER=2
DGEVOLUTION  += -DLATFIELD2_DEBUG_CUDA_SYNC
DGEVOLUTION  += -DNOTGH
#DGEVOLUTION  += -DPCL_EXTRA_CAPACITY=8388608
#DGEVOLUTION  += -DDEBUG_ALIGNMENT

# further compiler options
OPT          := -O3 -std=c++17 -g -ccbin mpic++ -arch=sm_80 --extended-lambda -Xcompiler -fopenmp
OPT_GCC	     := -O3 -std=c++17 -g -fopenmp

$(EXEC): $(SOURCE) $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB)

unit-tests: unit_tests.cu $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB) -DGADGET_LENGTH_CONVERSION=1 -DGADGET_VELOCITY_CONVERSION=1
	
lccat: lccat.cpp
	g++ $< -o $@ $(OPT_GCC) $(DGEVOLUTION) $(INCLUDE)
	
lcmap: lcmap.cpp
	g++ $< -o $@ $(OPT_GCC) $(DGEVOLUTION) $(INCLUDE) $(LIB) $(HPXCXXLIB)

express-setup: express_setup.cu $(HEADERS) makefile
	$(COMPILER) $< -o $@ $(OPT) $(DLATFIELD2) $(DGEVOLUTION) $(INCLUDE) $(LIB)

run-tests: unit-tests
	rm -f test_output_*
	srun -N 1 -n 4 -C gpu -A sm97 --time=5:00 --partition=debug ./unit-tests -n 2 -m 2 -Ngrid 128 -Npcl 2097152 -bench 8

run: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/adamek/testing/.
	export OMP_NUM_THREADS=72
	export OMP_PLACES=cores
	srun -N 1 -n 4 --cpus-per-task=72 -C gpu -A sm97 --time=5:00 --partition=debug --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh /capstor/scratch/cscs/adamek/testing/$(EXEC) -n 2 -m 2 -s /capstor/scratch/cscs/adamek/testing/settings.ini -p /capstor/scratch/cscs/adamek/testing/test.pre

profile: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/adamek/testing/.
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/users/adamek/local_arm/lib
	export OMP_NUM_THREADS=36
	export OMP_PLACES=cores
	srun -N 2 -n 16 --cpus-per-task=36 -C gpu -A sm97 --time=12:00 --partition=debug --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh ./nsys_wrapper.sh /capstor/scratch/cscs/adamek/testing/$(EXEC) -n 4 -m 4 -s /capstor/scratch/cscs/adamek/testing/benchmark.ini

scaling-test: $(EXEC)
	rsync -av ./$(EXEC) /capstor/scratch/cscs/adamek/testing/.
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/users/adamek/local_arm/lib
	export OMP_NUM_THREADS=72
	export OMP_PLACES=cores
	srun -N 9 -n 36 --cpus-per-task=72 -C gpu -A sm97 --time=19:00 --mem=192G --partition=normal --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh ./nsys_wrapper.sh /capstor/scratch/cscs/adamek/testing/$(EXEC) -n 6 -m 6 -s /capstor/scratch/cscs/adamek/testing/scaling-test.ini

clean:
	-rm -f $(EXEC) lccat lcmap unit-tests

