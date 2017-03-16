cmake_minimum_required(VERSION 2.8)
project(affineDSC)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

set(SOURCES main.cpp
    src/kpMatcher.cpp
    src/reconstructor.cpp
    epfl/Camera.cpp
    epfl/Mesh/LaplacianMesh.cpp
    epfl/Mesh/TriangleMesh.cpp
    epfl/Linear/LinearAlgebraUtils.cpp
    epfl/Linear/ObjectiveFunction.cpp
    epfl/Linear/IneqConstrFunction.cpp
    epfl/Linear/IneqConstrOptimize.cpp)

set(HEADER_FILES
    include/kpMatcher.h
    include/utilities.h
    include/reconstructor.h
    epfl/Camera.h
    epfl/Mesh/LaplacianMesh.h
    epfl/Mesh/TriangleMesh.h
    epfl/Linear/LinearAlgebraUtils.h
    epfl/Linear/ObjectiveFunction.h
    epfl/Linear/IneqConstrFunction.h
    epfl/Linear/IneqConstrOptimize.h)

set (CMAKE_CXX_FLAGS "-O2 -march=native" )

set ( DEPENDENCIES ${OpenCV_LIBS} ${OPENGL_LIBRARIES} )

# Find glfw header
find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
    PATHS /usr/X11R6/include
          /usr/include/X11
          /opt/graphics/OpenGL/include
          /opt/graphics/OpenGL/contrib/libglfw
          /usr/local/include
          /usr/include/GL
          /usr/include
)
# Find glfw library
find_library(GLFW_LIBRARIES NAMES glfw glfw3
        PATHS /usr/lib64
              /usr/lib
              /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /usr/local/lib64
              /usr/local/lib
              /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /usr/X11R6/lib
)

# Find armadillo
find_library( armadillo NAMES "armadillo"
        PATHS /usr/local/lib/ )

list(APPEND DEPENDENCIES m  ${armadillo} ${GLFW_LIBRARIES} )
include_directories(${GLFW_INCLUDE_DIR})


add_executable( affineDSC ${SOURCES} ${HEADER_FILES} )
target_link_libraries( affineDSC ${DEPENDENCIES} )

