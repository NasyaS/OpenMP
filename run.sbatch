#!/bin/bash
#SBATCH --job-name=dirichle_sadykova
#SBATCH --output=result.csv
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=20
#
#SBATCH --time=40:00
#SBATCH --mem-per-cpu=100

for omp_threads in 1 2 4 6 8 10 13 15 18 20
do

export OMP_NUM_THREADS=$omp_threads

for dim in 99 299 699 999
do
	for ((i=1; i <=2; i++))
	do
	./main $dim >cl_results${omp_threads}_${dim}_${i}.csv
	done
done

done