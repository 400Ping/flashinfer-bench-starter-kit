#!/bin/bash
# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Use paths relative to the uv environment
VENV_PATH="$PROJECT_ROOT/.venv"
NVCC=nvcc
TVM_FFI_PATH=$VENV_PATH/lib/python3.12/site-packages/tvm_ffi

# Check if TVM_FFI_PATH exists, if not, try to find it via python
if [ ! -d "$TVM_FFI_PATH" ]; then
    TVM_FFI_PATH=$(python3 -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))" 2>/dev/null)
fi

echo "Building MoE FFI library..."
$NVCC -shared -Xcompiler -fPIC \
    -arch=sm_90 \
    -I$TVM_FFI_PATH/include \
    -L$TVM_FFI_PATH/lib \
    -ltvm_ffi \
    -I$PROJECT_ROOT/solution/cuda/moe_expert_mlp \
    -o $SCRIPT_DIR/librouter_ffi.so \
    $PROJECT_ROOT/solution/cuda/moe_ffi.cu

echo "Building MoE Expert MLP tests..."
make -C $PROJECT_ROOT/solution/cuda/moe_expert_mlp CUDA_ARCH=sm_90 build-fallback

