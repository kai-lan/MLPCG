> [!NOTE]
> For running tests for our paper, please direct to `icml2024` branch.

# MLPCG

Machine Learned Preconditioned Conjugate Gradient (MLPCG) methods invents a compact linear neural network that approximate an inverse of a discrete Poisson matrix, and is used as a preconditioner for solving the Poisson equations. The Poisson equations arise from fluid simulation, where both fluid and solid object are present in the domain.
This project studies machine learning approach to accelerate the pressure Poisson solver in fluid simulations. Previous related work include [DeepGradient](https://arxiv.org/pdf/2205.10763.pdf) and [FluidNet](https://arxiv.org/pdf/1607.03597.pdf).


## Requirements and dependencies
* Python
* Pytorch: > 2.0
* AMGCL: https://github.com/ddemidov/amgcl
    * Boost: required by AMGCL
We recommend using virtual environment such as conda.

### Training dataset


#### For developers
---

The dataset can be generated from [tgsl](https://gitlab.com/teran-group/tgsl). Inside the `projects/incompressible_flow`.

The output includes `*.bgeo` files for visualization in Houdini, `flags_*.bin` for integer-valued images, `div_v_star_*.bin` rhs for pressure equation, `pressure_*.bin` solution, `A_*.bin` sparse matrix.

Your data should has the following path:
```
{PROJECT_FOLDER}/data/{scene}/{A_*.bin, div_v_star_*.bin, flags_*.bin, pressure_*.bin, *.bgeo}
{PROJECT_FOLDER}/data/{scene}/preprocessed/{rhs.pt, A.pt, fluid_cells.pt, b_*.pt}
```

To generate ritz vectors, modify and run `lib/create_dataset.py`. To generate training rhs, modify and run `preprocess.py`.

#### For users
---
Download the corresponding data from [here](https://drive.google.com/drive/folders/1q1D5LJmQqfNcJUDj5x3tC5cpIyRoSyGR?usp=drive_link). For 3D 256 training, we use the following data:
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

