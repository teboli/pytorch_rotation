from setuptools import setup, find_packages

__version__ = "0.1.0"
URL = "https://github.com/teboli/pytorch_rotation"

setup(
    name="torch_rotation",
    version=__version__,
    description="A Python package for rotating tensors using PyTorch",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    url=URL,
    keywords=[
        'pytorch',
        'rotation',
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch>=1.0.0",
    ],
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
