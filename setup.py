from setuptools import setup, find_packages

setup(
    name="deeplearning-models",
    version="0.1.0",
    description="PyTorch implementations of classic deep learning models",
    author="DeepLearning Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "einops>=0.4.1",
        "timm>=0.6.7",
        "pillow>=8.3.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.9.0",
        "matplotlib>=3.5.0",
    ],
)