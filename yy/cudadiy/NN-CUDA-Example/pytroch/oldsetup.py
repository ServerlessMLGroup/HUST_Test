from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


##this is what the old author use to create the .so, i modified this
setup(
    name="add2",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "add2",
            ["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)