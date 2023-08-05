cmake_minimum_required(VERSION 3.26)
project(test)

set(CMAKE_CXX_STANDARD 14)

# prepare libtorch env.
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

include_directories(
        ${TORCH_INCLUDE_DIRS}
        /usr/local/cuda/include
)

add_executable(test main.cpp)
target_link_directories(test PRIVATE /usr/local/cuda/lib64 /usr/local/libtorch/lib)
target_link_libraries(test ${TORCH_LIBRARIES} cublas cudart)