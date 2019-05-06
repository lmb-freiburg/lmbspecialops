#
# Provide options to configure the code generation options for the nvcc compiler
#

include( FindCUDA )

if( DEFINED ENV{CUDA_ARCH_LIST} )
    set( CUDA_ARCH_LIST $ENV{CUDA_ARCH_LIST} )
else()
    set( CUDA_ARCH_LIST Auto )
endif()
set( CUDA_ARCH_LIST ${CUDA_ARCH_LIST} CACHE LIST
        "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)." )

cuda_select_nvcc_arch_flags( CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST} )
message( STATUS "CUDA code generation flags: ${CUDA_ARCH_FLAGS}" )
