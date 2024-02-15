from setuptools import setup, find_packages

setup(
    name="auto_art",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "adversarial-robustness-toolbox>=1.13.0",
        "scikit-learn>=0.24.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "tqdm>=4.50.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "mypy>=0.800"
        ]
    }
) 