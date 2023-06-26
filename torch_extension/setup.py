from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='smblock',
#     ext_modules=[
#         CUDAExtension('smblock', [
#             'sm_block.cpp',
#             'sm_block_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

setup(
    name='smblock3d',
    ext_modules=[
        CUDAExtension('smblock3d', [
            'sm_block_3d.cpp',
            'sm_block_3d_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)