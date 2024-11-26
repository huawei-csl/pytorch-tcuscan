.PHONY: all build test

all: build test

build: build.sh add_custom.cpp pybind11.cpp
	pip3 install pybind11
	./build.sh

test: build test_add_custom.py
	export LD_LIBRARY_PATH=$(pwd)/build:${LD_LIBRARY_PATH} 
	pytest test_add_custom.py

