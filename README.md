# lmbspecialops

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

lmbspecialops is a collection of tensorflow ops.
The ops focus on networks for predicting depth and camera motion as in DeMoN, but many can also be useful for other tasks.

If you use this code for research please cite:
   
    @InProceedings{UZUMIDB17,
      author       = "B. Ummenhofer and H. Zhou and J. Uhrig and N. Mayer and E. Ilg and A. Dosovitskiy and T. Brox",
      title        = "DeMoN: Depth and Motion Network for Learning Monocular Stereo",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = " ",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/UZUMIDB17"
    }


See the [Op documentation](doc/lmbspecialops_doc.md) for a description of all functions.



## Requirements

Building and using lmbspecialops depends on the following libraries and programs

    tensorflow 1.4.0
    cmake 3.8
    python 3.5
    cuda 8.0.61 (required for building with gpu support)

The versions match the configuration we have tested on an ubuntu 16.04 system.
lmbspecialops can work with newer versions of the aforementioned dependencies, but this is not well tested.


## Installation instructions

Checkout the repository and run `setup.py install`.

```bash
git clone https://github.com/lmb-freiburg/lmbspecialops.git
cd lmbspecialops
pip install cmake-setuptools
python setup.py install
```

NVCC's CUDA gencode parameters are configured automatically based on available GPU devices.

If necessary, you can use the `CUDA_ARCH_LIST` environment variable or CMake variable to
generate code for specific architectures or compute capabilities instead.
`CUDA_ARCH_LIST` accepts a list of architecture names (e.g. Pascal, Volta, etc) and/or
compute capability versions (6.1, 7.0, etc).

To pass any parameters to CMake set the `CMAKE_COMMON_VARIABLES` environment variable. For example:

```bash
export CUDA_ARCH_LIST="Pascal 7.0"
export CMAKE_COMMON_VARIABLES="-DBUILD_WITH_CUDA=OFF" # to disable gpu support
python setup.py install
```

Alternatively, you can use `ccmake` or `cmake-gui` inside the created build directory
to configure any CMake variables.

```bash
python setup.py build_ext
cd build/temp.*
ccmake .
make clean
cd ../..
python setup.py install
```

After installation you can import and use the ops like this 

```python
import lmbspecialops
import tensorflow as tf
import numpy as np

tf.InteractiveSession()

A = tf.constant([1,2,np.nan])
B = lmbspecialops.replace_nonfinite(A)
print(B.eval()) # prints [1, 2, 0]
```

## Development

For development, it is preferable to use CMake directly without running `setup.py`.
```bash
mkdir build
cd build
cmake ..
# cmake .. -DBUILD_WITH_CUDA=OFF # to disable gpu support
make
```

To use the ops, you need to add the `python` directory to your python path.

Use `make test` in the build directory to run tests.

## License

lmbspecialops is under the [GNU General Public License v3.0](LICENSE.txt)

