#
# Get build flags from tensorflow to build custom ops correctly
#

find_package( PythonInterp REQUIRED )

execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include(), end='')"
        OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR )
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')"
        OUTPUT_VARIABLE TENSORFLOW_LIB_DIR )
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; \
        print(';'.join(flag[2:] for flag in tf.sysconfig.get_compile_flags() if flag.startswith('-D')), end='')"
        OUTPUT_VARIABLE TENSORFLOW_COMPILE_DEFINITIONS )

find_library( TENSORFLOW_FRAMEWORK_LIB tensorflow_framework PATHS "${TENSORFLOW_LIB_DIR}" NO_DEFAULT_PATH )

message( STATUS "${TENSORFLOW_INCLUDE_DIR}" )
message( STATUS "${TENSORFLOW_LIB_DIR}" )
message( STATUS "${TENSORFLOW_FRAMEWORK_LIB}" )
message( STATUS "TensorFlow compile definitions: ${TENSORFLOW_COMPILE_DEFINITIONS}" )
