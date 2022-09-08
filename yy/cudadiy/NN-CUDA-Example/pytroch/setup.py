from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


##this is the new one
setup(
    name="createContext",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "createContext",
            ["pytorch/crContext_ops.cpp", "kernel/createContext_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
