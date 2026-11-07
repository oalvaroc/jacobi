find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(serial serial.cpp)
target_link_libraries(serial PRIVATE OpenMP::OpenMP_CXX)

add_executable(omp omp.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)

add_executable(omp_simd omp_simd.cpp)
target_link_libraries(omp_simd PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(omp_simd PRIVATE -march=native)
