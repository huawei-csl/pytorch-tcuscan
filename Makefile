.PHONY: all clean setup_once build test

all: build test

clean:
	rm -rf build/ __pycache__/
	rm add_custom*.so

setup_ci:
	pip3 install --cache-dir=/scratch/TCUSCAN/wheels/ -r requirements.txt

# Setup a virtualenv first
# python3 -m venv venv
setup_once:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.4.0/torch_npu-2.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	pip install torch_npu-2.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --index-url https://download.pytorch.org/whl/cpu

setup_once_aarch64:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.4.0/torch_npu-2.4.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
	pip install torch_npu-2.4.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --index-url https://download.pytorch.org/whl/cpu

build: build.sh add_custom.cpp pybind11.cpp
	./build.sh -v ASCEND910B2

test: test_add_custom.py
	python3 -m pytest

profile: profile_add_custom.py
	python profile_add_custom.py --bench vadd
