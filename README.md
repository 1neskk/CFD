# CFD Engine

A high-performance Computational Fluid Dynamics (CFD) simulation engine built with C++, CUDA, and Vulkan. This project utilizes the Lattice Boltzmann Method (LBM) for fluid simulation and provides real-time visualization.

## Features

- **Lattice Boltzmann Method (LBM):** 3D fluid simulation using the D3Q19 model.
- **High Performance:** Accelerated using CUDA for parallel computation on the GPU.
- **Real-time Visualization:** Vulkan-based rendering with zero-copy CUDA-Vulkan interoperability.
- **Interactive:** Real-time camera controls and ImGui-based user interface.
- **Solid Boundaries:** Support for solid obstacles within the simulation domain.

## Prerequisites

Before building the project, ensure you have the following installed:

- **C++ Compiler:** Compatible with C++20 (e.g., GCC, Clang, MSVC).
- **CMake:** Version 3.24 or higher.
- **Make:** Build tool.
- **CUDA Toolkit:** For compiling the LBM solver kernels.
- **Vulkan SDK:** For the rendering engine.

## Build Instructions

The project includes a `Makefile` to simplify the build process.

### Release Build
To build the project in Release mode (optimized):
```bash
make build
```

### Debug Build
To build the project in Debug mode:
```bash
make debug
```

### Clean
To clean the build artifacts:
```bash
make clean
```

### Format Code
To format the source code using `clang-format`:
```bash
make format
```

### Run Tests
To build and run the tests:
```bash
make test
```

## Usage

After building, the executable will be located in the `build` directory.

```bash
./build/Engine/Engine
```

### Controls

- **W / S**: Move Camera Forward / Backward
- **A / D**: Move Camera Left / Right
- **Q / E**: Move Camera Down / Up
- **Right Mouse Button + Drag**: Rotate Camera (Look around)

## Project Structure

- **Core/**: Contains core engine components such as Application, Window, Input, Timer, and Logger.
- **Engine/**: Contains the simulation and rendering logic.
    - **Graphics/**: Vulkan renderer, Camera, and Shader management.
    - **Physics/**: LBM solver implementation using CUDA.
- **thirdparty/**: External libraries and dependencies.

## License

This project is licensed under the MIT License.
