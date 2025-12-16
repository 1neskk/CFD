cmake_minimum_required(VERSION 3.24)
project(Engine CUDA CXX)

include(FetchContent)
find_package(Vulkan REQUIRED)
find_package(OpenMP)

if(CMAKE_BUILD_TYPE STREQUAL "Test")
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/spdlog/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/tinyobjloader)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${Vulkan_INCLUDE_DIRS})

file(GLOB ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Physics/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Physics/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Graphics/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Graphics/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/*.cpp
)

file(GLOB ENGINE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Physics/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Physics/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Graphics/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/*.h
)

add_executable(${PROJECT_NAME}
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    Core
    ${CUDA_LIBRARIES}
    ${Vulkan_LIBRARIES}
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Relwithdebinfo" CACHE STRING "" FORCE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Relwithdebinfo")
    message(STATUS "Release with debug info build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -g)
else()
    message(FATAL_ERROR "Invalid build type. Use Debug or Release.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Test")
    enable_testing()
    add_subdirectory(tests)
endif()
