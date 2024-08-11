from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set environment variables for GCC 11
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'

def get_extensions():
    extensions = []

    # Define the paths to your CUDA source files and headers
    include_dirs = [
        os.path.join(os.getcwd(), "xlstm", "blocks", "slstm", "src", "cuda"),
        os.path.join(os.getcwd(), "xlstm", "blocks", "slstm", "src", "util")
    ]

    # Define the CUDA extension
    extensions.append(
        CUDAExtension(
            name='xlstm.slstm',
            sources=[
                'xlstm/blocks/slstm/src/cuda/slstm_backward.cu',
                'xlstm/blocks/slstm/src/cuda/slstm_backward_cut.cu',
                'xlstm/blocks/slstm/src/cuda/slstm_forward.cu',
                'xlstm/blocks/slstm/src/cuda/slstm_pointwise.cu',
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-g', '-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-Xptxas=-v',
                    '--use_fast_math'
                ]
            }
        )
    )

    return extensions

setup(
    name='xlstm',
    version='1.0.3',
    author='Maximilian Beck, Korbinian Poeppel, Andreas Auer',
    author_email='beck@ml.jku.at, poeppel@ml.jku.at, auer@ml.jku.at',
    description='A novel LSTM variant with promising performance compared to Transformers or State Space Models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NX-AI/xlstm',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.18.0',
        'omegaconf>=2.0.6',
        'dacite>=1.6.0'
    ],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={
        '': ['blocks/slstm/src/cuda/*', 'blocks/slstm/src/util/*']
    }
)
