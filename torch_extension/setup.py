# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(name='lltm_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='smblock',
    ext_modules=[
        CUDAExtension('smblock', [
            'sm_block.cpp',
            'sm_block_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })