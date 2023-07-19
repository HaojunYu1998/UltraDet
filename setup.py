import torch
from setuptools import find_packages, setup


setup(
    name="ultra_det",
    version="0.1dev0",
    author="Haojun Yu",
    description="Detection for ultrasound videos.",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=["numpy", "pillow", "pathspec"],
)
