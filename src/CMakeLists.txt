find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(serial serial.cpp)
target_link_libraries(serial PRIVATE OpenMP::OpenMP_CXX)
