cmake_minimum_required(VERSION 3.18)

project(LicensePlateRecognitionGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(LPR_SOURCES
    src/host/Main.cpp
    src/host/ImageIO.cpp
    src/host/PipelineContext.cpp
    src/host/SceneAnalyzer.cpp
    src/host/PlateRecognizer.cpp
    src/device/Preprocess.cu
    src/device/CCL.cu
    src/device/ImageOps.cu
    src/device/OtsuThreshold.cu
    src/device/KNN.cu
)

add_executable(${PROJECT_NAME} ${LPR_SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/host
    ${CMAKE_CURRENT_SOURCE_DIR}/include/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set; using native detection.")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(LPR_ENABLE_CUDA_LINEINFO OR NOT DEFINED LPR_ENABLE_CUDA_LINEINFO)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

message(STATUS "Build Configuration:")
message(STATUS "  C++ compiler:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CUDA compiler:      ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ─── Copy KNN binary data next to the binary ──────────────────────────────────
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/knn_data.bin
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# ─── clean-outputs: remove program-generated files (not profiling reports) ────
# Usage: cmake --build build --target clean-outputs
add_custom_target(clean-outputs
    COMMAND find "${CMAKE_SOURCE_DIR}" -maxdepth 1 -name "*.csv" -delete -print
    COMMENT "Removing generated CSV output files"
    VERBATIM
)
