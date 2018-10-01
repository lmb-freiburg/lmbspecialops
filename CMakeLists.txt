cmake_minimum_required( VERSION 3.5.1 )

project(lmbspecialops)
enable_testing()

if( NOT CMAKE_BUILD_TYPE )
        set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE )
endif()

option( BUILD_WITH_CUDA "If enabled builds cuda implementation of ops"  ON )

# enable all warnings
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_GLIBCXX_USE_CXX11_ABI=0" )
find_package( PythonLibs 3.6 REQUIRED )
add_subdirectory( src lib ) 
add_subdirectory( doc ) 
add_subdirectory( test test ) 

