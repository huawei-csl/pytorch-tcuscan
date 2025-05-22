#!/bin/bash
#DATAPATH=/storage/datasets/graphs-and-sparse-matrices/
DATAPATH=/home/gsorrentino/.ssgetpy/MM/matrices
#DATAPATH=~/datasets
#DATAPATH=/home/yzelman/Documents/datasets/graphs-and-matrices/
#ALPINSTALLPATH=/home/yzelman/graphblas/install-nolpf-kunpeng920/
#ALPINSTALLPATH=/home/yzelman/graphblas/install-nolpf-x86/
DATASETS=(ASIC_680k  cant  circuit5M  conf5_4-8x8-05  conf5_4-8x8-10  mac_econ_fwd500  mc2depi  mip1  pdb1HYS  rma10)
VECINNERREPS=(250 100 1 1 1)
OMPVECINNERREPS=(500 110 1 1 1)
OUTERREPS=30
OMP_FLAGS=OMP_PROC_BIND=true

echo "Entry date:"
date
echo " "

echo "Script contents:"
cat trigger.sh
echo " "

echo "Rebuilding ALP/GraphBLAS and SpMV, SpMSpV, and SpMSpM benchmarks:"
rm -r build install
mkdir build
cd build || exit
../bootstrap.sh --prefix=../install
make -j
make test_driver_spmv_reference
make test_driver_spmv_reference_omp
make test_driver_spmv_nonblocking
make test_driver_spmv_hyperdags
echo " "

export ${OMP_FLAGS?}
echo "OpenMP configuration:"
if [ -z "${OMP_NUM_THREADS}" ]; then
	        echo " - OMP_NUM_THREADS is not set or has no value"
	else
		        echo " - OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi
if [ -z "${OMP_PLACES}" ]; then
	        echo " - OMP_PLACES is not set or has no value"
	else
		        echo " - OMP_PLACES=${OMP_PLACES}"
fi
if [ -z "${OMP_PROC_BIND}" ]; then
	        echo "Warning: OMP_PROC_BIND is not set or has no value."
		        echo "         This is probably not what you want!"
		else
			        echo " - OMP_PROC_BIND=${OMP_PROC_BIND}"
fi
echo " "

echo "SpMV, sequential:"
for i in "${!DATASETS[@]}"
do
	dataset=${DATASETS[i]}
	inner=${VECINNERREPS[i]}
	file=${DATAPATH}/${dataset}.mtx
	ls -lha "${file}"
        ./tests/performance/driver_spmv_reference "${file}" direct "${inner}" ${OUTERREPS} 2> "${dataset}".alp_spmv.vec
done
echo " "

echo "SpMV, hyperdags:"
for i in "${!DATASETS[@]}"
do
        dataset=${DATASETS[i]}
        inner=${VECINNERREPS[i]}
        file=${DATAPATH}/${dataset}.mtx
        ls -lha "${file}"
        ./tests/performance/driver_spmv_hyperdags "${file}" direct "${inner}" ${OUTERREPS} 2> "${dataset}".alp_spmv_hyperdags.vec
done
echo " "

echo "SpMV, OpenMP:"
for i in "${!DATASETS[@]}"
do
	dataset=${DATASETS[i]}
	inner=${OMPVECINNERREPS[i]}
	file=${DATAPATH}/${dataset}.mtx
	ls -lha "${file}"
	./tests/performance/driver_spmv_reference_omp "${file}" direct "${inner}" ${OUTERREPS} 2> "${dataset}".alp_spmv_omp.vec
done
echo " "

echo "SpMV, NonBlocking:"
for i in "${!DATASETS[@]}"
do
        dataset=${DATASETS[i]}
        inner=${OMPVECINNERREPS[i]}
        file=${DATAPATH}/${dataset}.mtx
        ls -lha "${file}"
        ./tests/performance/driver_spmv_nonblocking "${file}" direct "${inner}" ${OUTERREPS} 2> "${dataset}".alp_spmv_nonblocking.vec
done
echo " "


