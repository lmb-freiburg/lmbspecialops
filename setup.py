from cmake_setuptools import CMakeExtension, CMakeBuildExt
from setuptools import setup

setup(
    name='lmbspecialops',
    setup_requires=['cmake-setuptools', 'tensorflow'],
    ext_modules=[CMakeExtension('lmbspecialops')],
    cmdclass={'build_ext': CMakeBuildExt},
    package_dir={'': 'python'},
    packages=['lmbspecialops'],
    zip_safe=False,
)
