cmake_minimum_required(VERSION 2.8)
project(3DV)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCES main.cpp
    src/KpMatcher.cpp
    src/Reconstructor.cpp
    src/Utilities.cpp
    src/SimulatedAnnealing.cpp
    epfl/Camera.cpp
    epfl/Mesh/LaplacianMesh.cpp
    epfl/Mesh/TriangleMesh.cpp
    epfl/Linear/LinearAlgebraUtils.cpp
    epfl/Linear/ObjectiveFunction.cpp
    epfl/Linear/IneqConstrFunction.cpp
    epfl/Linear/IneqConstrOptimize.cpp)

set(HEADER_FILES
    include/KpMatcher.h
    include/Utilities.h
    include/Reconstructor.h
    include/Utilities.h
    include/SimulatedAnnealing.h
    include/Alghoritm.h
    epfl/Camera.h
    epfl/Mesh/LaplacianMesh.h
    epfl/Mesh/TriangleMesh.h
    epfl/Linear/LinearAlgebraUtils.h
    epfl/Linear/ObjectiveFunction.h
    epfl/Linear/IneqConstrFunction.h
    epfl/Linear/IneqConstrOptimize.h)

set (CMAKE_CXX_FLAGS "-O2 -march=native" )

set ( DEPENDENCIES ${OpenCV_LIBS} )

# Find armadillo
find_library( armadillo NAMES "armadillo"
        PATHS /usr/local/lib/ )

list(APPEND DEPENDENCIES m  ${armadillo} )


add_executable( 3DV ${SOURCES} ${HEADER_FILES} )
target_link_libraries( 3DV ${DEPENDENCIES} )

