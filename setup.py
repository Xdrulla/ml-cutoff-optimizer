from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-cutoff-optimizer",
    version="0.1.0",
    author="Luan Drulla",
    author_email="serighelli003@gmail.com",
    description="Professional toolkit for binary classification threshold optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xdrulla/ml-cutoff-optimizer",
    project_urls={
        "Bug Tracker": "https://github.com/Xdrulla/ml-cutoff-optimizer/issues",
        "Documentation": "https://github.com/Xdrulla/ml-cutoff-optimizer#readme",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "app": [
            "streamlit>=1.20.0",
        ],
        "docs": [
            "jupyterlab>=3.4.0",
            "notebook>=6.4.0",
        ],
    },
)
