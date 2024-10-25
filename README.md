# Instruct-Object2Object [Course Project of Computational Imaging @ UofT]

**Author: Yuyang Tang (yytang@gmail.com), Xutong Jiang (xjiang@cs.toronto.edu), Chenghao Gong**

**Our project poster is available [here](Instruct-Object2Object.pdf).**

## Installation

### 1. Install Nerfstudio dependencies

Instruct-Object2Object is based on Instruct-Nerf2Nerf, which is built on Nerfstudio and therefore has the same dependency reqirements. Specfically [PyTorch](https://pytorch.org/) and [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) are required.

Follow the instructions [at this link](https://docs.nerf.studio/quickstart/installation.html) to create the environment and install dependencies. Only follow the commands up to tinycudann. After the dependencies have been installed, return here.

### 2. Installing Instruct-Object2Object

```
cd instruct-object2object
pip install --upgrade pip setuptools
pip install -e .

# To confirm installation
ns-train -h
```

After that you can use method io2o to train datasets with masks!

### 3. Installing Segment-Anything
```
conda install -c conda-forge opencv
conda install -c conda-forge groundingdino-py
conda install -c conda-forge segment-anything
```
### 4. Installing Cargo and Texture-Synthesis
The inpainting part is run by cargo. Please follow the [instructions](https://doc.rust-lang.org/cargo/getting-started/installation.html) to build cargo environment. Then build texture-synthesis library from [here](https://github.com/EmbarkStudios/texture-synthesis).
