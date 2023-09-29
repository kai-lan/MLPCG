# A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions

This is the codebase for our paper.


## Requirements and dependencies
* Python
* Pytorch: > 2.0
We recommend using a virtual environment such as conda.



## Setup
* Clone AMGCL: https://github.com/ddemidov/amgcl into `cxx_src` folder.
    * Boost: required by AMGCL
* CUDA Tookit required: https://developer.nvidia.com/cuda-downloads



#### For users
---
Download the corresponding data from [here]. For 3D 256 training, we use the following data:
```
dambreak_N256_200_3D
dambreak_hill_N128_N256_200_3D
dambreak_dragons_N128_N256_200_3D
ball_cube_N256_200_3D
ball_bowl_N256_200_3D
standing_dipping_block_N256_200_3D
standing_rotating_blade_N256_200_3D
waterflow_pool_N256_200_3D
waterflow_panels_N256_200_3D
waterflow_rotating_cube_N256_200_3D
```

The ritz vectors (`ritz_1600.npy`) have been generated for you inside `preprocessed` folder except for `dambreak_hill_N128_N256_200_3D`, which includes all the `*.pt` vectors ready.

A trained model is also included in `trained_model` folder. Download it and place it in `{PROJECT_DIR}/output/output_3D_256`.

Run `python user_preprocess.py` to generate all the necessary `*.pt` files for training.

Run `python user_train_256.py` for training. Adjust the batch size to maximize GPU usage.

