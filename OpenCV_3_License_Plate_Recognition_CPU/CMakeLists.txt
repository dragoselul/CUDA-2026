cmake_minimum_required(VERSION 3.18)
project(LicensePlateRecognition)

# 1. Standard C++ Setup
set(CMAKE_CXX_STANDARD 17) # Upped to 17 for better modern library support
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find OpenCV
find_package(OpenCV REQUIRED)

# 3. Include Headers
include_directories(${OpenCV_INCLUDE_DIRS})

# 4. Define Source Files
set(SOURCE_FILES
        Main.cpp
        DetectChars.cpp
        DetectPlates.cpp
        PossibleChar.cpp
        PossiblePlate.cpp
        Preprocess.cpp
        Conio.h
)

# 5. Create the Executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 6. Link Libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# 7. Copy KNN training data next to the binary
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/classifications.xml
    ${CMAKE_CURRENT_SOURCE_DIR}/images.xml
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# 8. Output Build Info (Nice for debugging)
message(STATUS "Build Configuration:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID}")
