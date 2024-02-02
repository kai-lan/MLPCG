# A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions

This is the codebase for our paper.


## Requirements and dependencies
* Python.
* [Pytorch](https://pytorch.org/get-started/locally/) > 2.0.
* [CUDA Tookit](https://developer.nvidia.com/cuda-downloads) required.
* [Boost](https://www.boost.org/): required by AMGCL
* `git submodule update --init --recursive` should register the following submodules in `cxx_src` folder:
    * [AMGCL](https://github.com/ddemidov/amgcl).
    * [AMGX](https://github.com/NVIDIA/AMGX).
    * [VexCL](https://github.com/ddemidov/vexcl).
    * [PyBind11](https://github.com/pybind/pybind11).

* Download [eigen 3.4](https://eigen.tuxfamily.org/index.php?title=Main_Page) into the project folder, and name it `eigen-3.4.0`.
* [CuPy](https://cupy.dev/) for CUDA implementation of CG.
* [SciPy](https://scipy.org/) for linear algebra support.
* [Ninja](https://ninja-build.org/) required to load Torch C++ extension.

We recommend using a virtual environment such as `conda`.

# Setup
In order to test `AMGCL` or `IC`, you need to do the following inside `cxx_src` folder, :
```
    mkdir build && cd build
    cmake .. -GNinja
    ninja
```
In order to test `AMGX`, follow the instructions on [here](https://github.com/NVIDIA/AMGX) to build the project. Then inside `cxx_src/pyamgx`, run
```
pip install .
```

## Testing
Download test data and trained model from [here](https://drive.google.com/file/d/1HvPYeFbw34-esAd6Lk5LaQu4w2DuFUMq/view?usp=drive_link).

Inside the project folder, unzip it, and you should expect the following files:
```
data/dambreak_pillars_N128_N256_200_3D/div_v_star_200.bin
data/dambreak_pillars_N128_N256_200_3D/A_200.bin
data/dambreak_pillars_N128_N256_200_3D/flags_200.bin
data/smoke_bunny_N256_200_3D/div_v_star_200.bin
data/smoke_bunny_N256_200_3D/A_200.bin
data/smoke_bunny_N256_200_3D/flags_200.bin
data/smoke_solid_N128_200_3D/div_v_star_200.bin
data/smoke_solid_N128_200_3D/A_200.bin
data/smoke_solid_N128_200_3D/flags_200.bin
data/waterflow_ball_N256_200_3D/div_v_star_200.bin
data/waterflow_ball_N256_200_3D/A_200.bin
data/waterflow_ball_N256_200_3D/flags_200.bin
output/output_3D_128/checkpt_mixedBCs_M11_ritz1600_rhs800_l5_trilinear_25.tar
output/output_3D_128/checkpt_mixedBCs_M11_ritz1600_rhs800_l4_trilinear_62.tar
```

Run the test with
```
python test.py
```
The first time will be slow, as the PyTorch CUDA extension is compiled.
