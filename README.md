# MLPCG

This project studies machine learning approach to accelerate the pressure solver in fluid simulations. We aim to provide one model that is capable in both Dirichlet and Neumann BCs. The project is inspired by [DeepGradient](https://arxiv.org/pdf/2205.10763.pdf) and [FluidNet](https://arxiv.org/pdf/1607.03597.pdf).


## Requirements
* Python
* Pytorch

We recommend using virtual environment such as conda.

### Dataset
The dataset can be generated from tgsl [Kai's branch](https://gitlab.com/teran-group/tgsl/-/tree/Kai). Right now we are experimenting with 2D low-resolution (64, 128) examples.

