#!/bin/bash

CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE:-$0}")"
    pwd
)
cd "$CURRENT_DIR"

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend910B4"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH="$HOME"/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
# shellcheck source=/dev/null
source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash
echo "Current compile soc version is ${SOC_VERSION}"

CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
export CMAKE_PREFIX_PATH

# TORCH_NPU_PATH is the location where PyTorch Ascend Adapter (torch_npu) is installed.
TORCH_NPU_PATH=$(python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))")
export TORCH_NPU_PATH

echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
echo "TORCH_NPU_PATH=${TORCH_NPU_PATH}"



set -e

export CMAKE_GENERATOR="Unix Makefiles"
pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu \
                 --config-settings=cmake.define.TORCH_NPU_PATH="${TORCH_NPU_PATH}"
