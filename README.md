# MLPCG

This project studies machine learning approach to accelerate the pressure Poisson solver in fluid simulations. We aim to provide one model that is capable in both Dirichlet and Neumann BCs. The project is inspired by [DeepGradient](https://arxiv.org/pdf/2205.10763.pdf) and [FluidNet](https://arxiv.org/pdf/1607.03597.pdf).


## Requirements
* Python
* Pytorch

We recommend using virtual environment such as conda.

### Dataset
The dataset can be generated from [tgsl](https://gitlab.com/teran-group/tgsl). Inside the `projects/incompressible_flow`. Run the following command:
```
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 ..
make -j
```
There are multiple scenes youcan try:
- dambreak: `./incompressible_flow --example 60 --frames_per_second 60  -N 256  --setup_id 1 --final_frame 200 --max_dt .003 --cfl 1 --max_cg_iterations 5000 --cg_tol 1e-5 --flip_ratio 0.99 --output_dir output_dambreak_N256_200`
- Two dropping circles: `./incompressible_flow --example 60 --frames_per_second 60  -N 256  --setup_id 2 --final_frame 200 --max_dt .003 --cfl 1 --max_cg_iterations 5000 --cg_tol 1e-5 --flip_ratio 0.99 --output_dir output_circles_N256_200`
- Circle dropping into pool: `./incompressible_flow --example 60 --frames_per_second 60  -N 256  --setup_id 3 --final_frame 200 --max_dt .003 --cfl 1 --max_cg_iterations 5000 --cg_tol 1e-5 --flip_ratio 0.99 --output_dir output_circlepool_N256_200`
- Tilt water in container: `./incompressible_flow --example 60 --frames_per_second 60  -N 256  --setup_id 4 --final_frame 200 --max_dt .003 --cfl 1 --max_cg_iterations 5000 --cg_tol 1e-5 --flip_ratio 0.99 --output_dir output_wedge_N256_200`

The output includes `*.bgeo` files for visualization in Houdini, `flags_*.bin` for integer-valued images, `div_v_star_*.bin` rhs for pressure equation, `pressure_*.bin` solution, `A_*.bin` sparse matrix.

To generate ritz vectors, modify and run `lib/create_dataset.py`. To generate training rhs, modify and run `preprocess.py`.

Right now we are mainly experimenting with 2D low-resolution `64 x 64 ~ 256 x 256` examples.

