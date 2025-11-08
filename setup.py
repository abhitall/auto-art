from setuptools import setup, find_packages

setup(
    name="auto_art",
    version="0.1.0",
    description="Automated Adversarial Robustness Testing Framework",
    author="Auto-ART Team",
    python_requires=">=3.8,<4.0",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        # Core dependencies
        "numpy>=1.19.0,<2.0.0",
        "adversarial-robustness-toolbox>=1.13.0,<2.0.0",
        "scikit-learn>=0.24.0,<2.0.0",
        "opencv-python>=4.5.0,<5.0.0",
        "matplotlib>=3.3.0,<4.0.0",
        "pandas>=1.2.0,<3.0.0",
        "tqdm>=4.50.0,<5.0.0",
        # API dependencies
        "Flask>=2.0.0,<4.0.0",
    ],
    extras_require={
        # Deep learning frameworks (optional - install based on need)
        "pytorch": [
            "torch>=1.7.0,<3.0.0",
        ],
        "tensorflow": [
            "tensorflow>=2.4.0,<3.0.0",  # or tensorflow-cpu for CPU-only
        ],
        "tensorflow-cpu": [
            "tensorflow-cpu>=2.4.0,<3.0.0",
        ],
        "all-frameworks": [
            "torch>=1.7.0,<3.0.0",
            "tensorflow>=2.4.0,<3.0.0",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.0.0,<9.0.0",
            "pytest-cov>=4.0.0,<6.0.0",
            "pytest-mock>=3.0.0,<4.0.0",
            "black>=23.0.0,<25.0.0",
            "isort>=5.0.0,<6.0.0",
            "mypy>=0.800,<2.0.0",
            "flake8>=4.0.0,<8.0.0",
        ],
        # Security testing
        "security": [
            "bandit>=1.7.0,<2.0.0",
            "safety>=2.0.0,<4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
) 