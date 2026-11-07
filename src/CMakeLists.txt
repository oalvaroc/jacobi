find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(serial serial.cpp)
target_link_libraries(serial PRIVATE OpenMP::OpenMP_CXX)

add_executable(omp omp.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)

add_executable(omp_simd omp_simd.cpp)
target_link_libraries(omp_simd PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(omp_simd PRIVATE -march=native)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(cuda cuda.cpp cuda_jacobi.cu)
    set_target_properties(cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "CUDA compiler not found, skipping the cuda target")
endif()
