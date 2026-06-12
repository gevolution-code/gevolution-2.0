# CSCS Alps execution helpers. Included only by config/alps.mk.

.PHONY: run-tests run profile scaling-test

run-tests: unit-tests
	rm -f test_output_*
	srun -N 1 -n 4 -C gpu -A $(ALPS_ACCOUNT) --time=5:00 --partition=$(ALPS_PARTITION) ./unit-tests -n 2 -m 2 -Ngrid 128 -Npcl 2097152 -bench 8

run: $(EXEC)
	rsync -av ./$(EXEC) $(ALPS_SCRATCH)/.
	OMP_NUM_THREADS=$(ALPS_THREADS) OMP_PLACES=cores srun -N 1 -n 4 --cpus-per-task=$(ALPS_THREADS) -C gpu -A $(ALPS_ACCOUNT) --time=5:00 --partition=$(ALPS_PARTITION) --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh $(ALPS_SCRATCH)/$(EXEC) -n 2 -m 2 -s $(ALPS_SCRATCH)/settings.ini -p $(ALPS_SCRATCH)/test.pre

profile: $(EXEC)
	rsync -av ./$(EXEC) $(ALPS_SCRATCH)/.
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/users/adamek/local_arm/lib OMP_NUM_THREADS=$(ALPS_THREADS) OMP_PLACES=cores srun -N 1 -n 4 --cpus-per-task=$(ALPS_THREADS) -C gpu -A $(ALPS_PROFILE_ACCOUNT) --time=12:00 --partition=$(ALPS_PARTITION) --hint=exclusive --cpu-bind=socket ./gpu-bind.sh ./nsys_wrapper.sh $(ALPS_SCRATCH)/$(EXEC) -n 2 -m 2 -s $(ALPS_SCRATCH)/sort-test.ini

scaling-test: $(EXEC)
	rsync -av ./$(EXEC) $(ALPS_SCRATCH)/.
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/users/adamek/local_arm/lib OMP_NUM_THREADS=$(ALPS_THREADS) OMP_PLACES=cores srun -N 9 -n 36 --cpus-per-task=$(ALPS_THREADS) -C gpu -A $(ALPS_ACCOUNT) --time=19:00 --mem=192G --partition=normal --hint=exclusive --cpu-bind=socket ./mps-wrapper.sh ./nsys_wrapper.sh $(ALPS_SCRATCH)/$(EXEC) -n 6 -m 6 -s $(ALPS_SCRATCH)/scaling-test.ini
