from cmake_setuptools import CMakeExtension, CMakeBuildExt
from setuptools import setup

setup(
    name='lmbspecialops',
    description='A python wrapper for tf to ease creation of network definitions',
    version='A collection of tensorflow ops',
    url='https://github.com/lmb-freiburg/lmbspecialops',
    license='GPLv3.0',
    author='lmb-freiburg',
    author_email='',
    python_requires='>=3.5',
    setup_requires=['cmake-setuptools', 'tensorflow'],
    ext_modules=[CMakeExtension('lmbspecialops')],
    cmdclass={'build_ext': CMakeBuildExt},
    package_dir={'': 'python'},
    packages=['lmbspecialops'],
    zip_safe=False,
)
