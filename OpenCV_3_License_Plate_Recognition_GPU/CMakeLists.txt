cmake_minimum_required(VERSION 3.18)

project(LicensePlateRecognitionGPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LPR_ENABLE_CUDA "Enable CUDA language and compile CUDA sources" ON)
option(LPR_ENABLE_CUDA_LINEINFO "Enable nvcc -lineinfo for CUDA profiling" ON)

find_package(OpenCV REQUIRED)

set(LPR_CPP_SOURCES
	Main.cpp
	DetectChars.cpp
	DetectPlates.cpp
	PossibleChar.cpp
	PossiblePlate.cpp
	Preprocess.cu
)

set(LPR_CUDA_SOURCES)

if(LPR_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)

	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)

		if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
			message(STATUS "CMAKE_CUDA_ARCHITECTURES not set; using compiler defaults. Set -DCMAKE_CUDA_ARCHITECTURES=<arch> for reproducible builds.")
		endif()

		file(GLOB LPR_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
	else()
		message(WARNING "CUDA compiler not found. Building CPU-only sources in GPU project.")
	endif()
endif()

add_executable(${PROJECT_NAME}
		Preprocess.cuh)

target_sources(${PROJECT_NAME}
	PRIVATE
	${LPR_CPP_SOURCES}
	${LPR_CUDA_SOURCES}
)

target_include_directories(${PROJECT_NAME}
	PRIVATE
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
	${OpenCV_LIBS}
)

if(CMAKE_CUDA_COMPILER AND LPR_CUDA_SOURCES)
	set_source_files_properties(${LPR_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

	if(LPR_ENABLE_CUDA_LINEINFO)
		target_compile_options(${PROJECT_NAME} PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
		)
	endif()
endif()

message(STATUS "Build Configuration:")
message(STATUS "  OpenCV version:        ${OpenCV_VERSION}")
message(STATUS "  C++ compiler:          ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CUDA requested:        ${LPR_ENABLE_CUDA}")
message(STATUS "  CUDA compiler found:   ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA lineinfo:         ${LPR_ENABLE_CUDA_LINEINFO}")
list(LENGTH LPR_CPP_SOURCES LPR_CPP_COUNT)
list(LENGTH LPR_CUDA_SOURCES LPR_CUDA_COUNT)
message(STATUS "  C++ sources:           ${LPR_CPP_COUNT}")
message(STATUS "  CUDA sources:          ${LPR_CUDA_COUNT}")

