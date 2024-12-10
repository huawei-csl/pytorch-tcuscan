.PHONY: all clean setup_once build test

all: build test

clean:
	rm -rf build/ __pycache__/
	rm add_custom*.so

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch==2.4.0+cpu  --index-url https://download.pytorch.org/whl/cpu
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.4.0/torch_npu-2.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	pip install torch_npu-2.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

build: build.sh add_custom.cpp pybind11.cpp
	./build.sh -v ASCEND910B4

test: test_add_custom.py
	pytest test_add_custom.py

