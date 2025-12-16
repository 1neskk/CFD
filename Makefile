.PHONY: all build debug relwithdebinfo clean format help
all: build

build: shaders
	mkdir -p build
	echo "Building project in Release mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release .. && \
	make

shaders:
	echo "Compiling shaders..."
	find Engine/src/Graphics/Shaders -name "*.vert" -o -name "*.frag" -o -name "*.comp" | xargs -I{} glslc {} -o {}.spv

debug: shaders
	mkdir -p build
	echo "Building project in Debug mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	make

relwithdebinfo:
	mkdir -p build
	echo "Building project in RelWithDebInfo mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Relwithdebinfo .. && \
	make

clean:
	rm -rf build

format:
	echo "Formatting code..."
	clang-format Core/src/*.h Core/src/*.cu Core/src/*.cpp Engine/src/Physics/*.h Engine/src/Physics/*.cpp Engine/src/Physics/*.cu Engine/src/Graphics/*.h Engine/src/Graphics/*.cpp Engine/src/Graphics/*.cu -i --style=file

help:
	@echo "Available targets:"
	@echo "  all: Build and run tests"
	@echo "  build: Build the project in release mode"
	@echo "  test: Build and run tests"
	@echo "  debug: Build the project in debug mode"

test:
	mkdir -p build
	echo "Building and running tests..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	make && \
	cd Engine/tests && \
	ctest --output-on-failure
