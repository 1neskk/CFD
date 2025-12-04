cmake_minimum_required(VERSION 3.24)

set(TEST_SOURCES
    test_lbm_solver.cu
    test_renderer.cu
    test_interop.cu
)

find_package(CUDAToolkit REQUIRED)
add_executable(EngineTests ${TEST_SOURCES})
target_link_libraries(EngineTests
    PRIVATE
    gtest
    gtest_main
    EngineLib
    Core
    ${Vulkan_LIBRARIES}
    ${CUDA_LIBRARIES}
)
target_include_directories(EngineTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/src
)

include(GoogleTest)
gtest_discover_tests(EngineTests)
