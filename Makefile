.PHONY: all setup_once build test

all: build test

setup_once:
	pip3 install pybind11 pyyaml
	pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu
	pip3 install https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

build: build.sh add_custom.cpp pybind11.cpp
	./build.sh -v ASCEND910B4

test: test_add_custom.py
	export LD_LIBRARY_PATH=$(pwd)/build:${LD_LIBRARY_PATH} 
	pytest test_add_custom.py

