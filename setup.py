#!/usr/bin/env python3
"""
Setup script for Breast Cancer Detection Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="breast-cancer-detection",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="A machine learning project for breast cancer detection using Decision Tree and Logistic Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/breast-cancer-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "breast-cancer-analysis=breast_cancer_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.ipynb"],
    },
)
