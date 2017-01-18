cmake_minimum_required(VERSION 2.8)
project(affineDSC)
find_package(OpenCV REQUIRED)
link_directories(${MYPROJECT_DIR})

include_directories(${MYPROJECT_DIR})

set(SOURCES main.cpp
            src/affineHelper.cpp
            src/utilities.cpp)

set(HEADER_FILES include/affineHelper.h
                 include/utilities.h)

set (CMAKE_CXX_FLAGS "-O2 -march=native")

add_executable(affineDSC ${SOURCES} ${HEADER_FILES})
target_link_libraries( affineDSC ${OpenCV_LIBS})
