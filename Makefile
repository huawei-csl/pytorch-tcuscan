.PHONY: all clean setup_ci setup_once setup_once_aarch64 build test profile

all: build test

clean:
	rm -rf build/ tests/__pycache__/
	rm -f *.so

setup_ci:
	pip3 install --cache-dir=/scratch/TCUSCAN/wheels/ -r requirements.txt

setup_once:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	pip3 install torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --index-url https://download.pytorch.org/whl/cpu

setup_once_aarch64:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
	pip3 install torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --index-url https://download.pytorch.org/whl/cpu

build: build.sh src/vadd.cpp src/diff.cpp src/seg_scan_single_core.cpp src/pybind11.cpp
	./build.sh -v ASCEND910B4

test:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/  python3 -m pytest tests/

profile_%: profile_$*_fp116 profile_$*_int16

profile_fp16_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench $* --dtype fp16

profile_int16_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench $* --dtype int16
