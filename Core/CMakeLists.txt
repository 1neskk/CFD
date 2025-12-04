cmake_minimum_required(VERSION 3.24)
project(Core CUDA CXX)

find_package(Vulkan REQUIRED)
find_package(X11 REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/spdlog/include)

set(IMGUI_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(ImGui STATIC
    ${IMGUI_SRC}
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw ${CMAKE_BINARY_DIR}/thirdparty/glfw)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
)

add_library(${PROJECT_NAME} STATIC
    ${SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

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

target_link_libraries(${PROJECT_NAME}
    ${Vulkan_LIBRARIES}
    ${CUDA_LIBRARIES}
    ImGui
    glfw
    ${X11_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
