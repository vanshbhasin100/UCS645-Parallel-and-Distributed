# ============================================================
# Lab5 - MPI Part II Makefile
# UCS645: Parallel & Distributed Computing
# ============================================================

CC     = mpicc
CFLAGS = -O3 -lm

TARGETS = daxpy broadcast_race dot_product primes perfect_numbers

all: $(TARGETS)

daxpy:
	$(CC) $(CFLAGS) -o daxpy daxpy.c

broadcast_race:
	$(CC) $(CFLAGS) -o broadcast_race broadcast_race.c

dot_product:
	$(CC) $(CFLAGS) -o dot_product dot_product.c

primes:
	$(CC) $(CFLAGS) -o primes primes.c -lm

perfect_numbers:
	$(CC) $(CFLAGS) -o perfect_numbers perfect_numbers.c

# ---- Run all with default process counts ----
run_all:
	@echo "=== Q1: DAXPY ==="
	mpirun -np 1 ./daxpy
	mpirun -np 2 ./daxpy
	mpirun -np 4 ./daxpy
	mpirun -np 8 ./daxpy
	@echo "=== Q2: Broadcast Race ==="
	mpirun -np 2 ./broadcast_race
	mpirun -np 4 ./broadcast_race
	mpirun -np 8 ./broadcast_race
	@echo "=== Q3: Dot Product ==="
	mpirun -np 1 ./dot_product
	mpirun -np 2 ./dot_product
	mpirun -np 4 ./dot_product
	mpirun -np 8 ./dot_product
	@echo "=== Q4: Primes ==="
	mpirun -np 4 ./primes
	@echo "=== Q5: Perfect Numbers ==="
	mpirun -np 4 ./perfect_numbers

clean:
	rm -f $(TARGETS)

.PHONY: all run_all clean
