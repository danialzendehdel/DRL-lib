from setuptools import setup, find_packages

setup(
    name="drl-lib",
    version="0.1.0",
    description="A Deep Reinforcement Learning library implementing SAC and other algorithms",
    author="Your Name",
    author_email="zendehdel.d@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "gymnasium>=0.26.0",
        "numpy>=1.19.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Homepage": "https://github.com/danialzendehdel/DRL-lib",
        "Bug Tracker": "https://github.com/danialzendehdel/DRL-lib/issues",
    },
) 