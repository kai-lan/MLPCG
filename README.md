# A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions

This is the codebase for our paper.


## Requirements and dependencies
* Python
* Pytorch: > 2.0
* CUDA Tookit required: https://developer.nvidia.com/cuda-downloads
* Clone `AMGCL`: https://github.com/ddemidov/amgcl into `cxx_src` folder.
    * Boost: required by AMGCL
    * In order to test `AMGCL` or `IC`, you need to do the following:
        ```
            mkdir build && cd build
            cmake ..
            make -j
        ```
* Clone `VexCL`: https://github.com/ddemidov/vexcl into `cxx_src` folder.
* Clone `PyBind11`: https://github.com/pybind/pybind11 into `cxx_src` folder.
* Download `eigen 3.4` from https://eigen.tuxfamily.org/index.php?title=Main_Page into the project folder, and name it `eigen-3.4.0`.
* CuPy: https://cupy.dev/
* SciPy

We recommend using a virtual environment such as conda.



### Testing
Download test data and trained model from [here](https://drive.google.com/file/d/1HvPYeFbw34-esAd6Lk5LaQu4w2DuFUMq/view?usp=drive_link).

Inside the project folder, unzip it, and you should expect the following files:
```
data/dambreak_pillars_N128_N256_200_3D/div_v_star_100.bin
data/dambreak_pillars_N128_N256_200_3D/A_100.bin
data/dambreak_pillars_N128_N256_200_3D/flags_100.bin
output/output_3D_128/checkpt_mixedBCs_M10_ritz1600_rhs800_imgs3_lr0.0001_30.tar
```

Run the test with
```
python test.py
```
The first time will be slow, as the PyTorch CUDA extension is compiled.
