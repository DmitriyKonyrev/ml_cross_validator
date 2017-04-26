cmake_minimum_required (VERSION 2.8.12)


if(CMAKE_VERSION VERSION_GREATER 3.0.0)
	cmake_policy(SET CMP0043 OLD)
endif()

IF (USE_CXX11_ABI)
    ADD_DEFINITIONS (-D_GLIBCXX_USE_CXX11_ABI=1)
ELSE()
    ADD_DEFINITIONS (-D_GLIBCXX_USE_CXX11_ABI=0)
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3 /MP /openmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP /openmp")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
project(machine_learning_methods CXX)

find_package(Boost REQUIRED QUIET)
FIND_PACKAGE(Boost COMPONENTS program_options system filesystem REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmakemodules")
set(CMAKE_BUILD_WITH_INSTALL_RPATH true)
set(CMAKE_INSTALL_RPATH ".")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(add_source_groups)
include(group_files)
include(filter_sources)
#include(Findvector_core)
#include(GenerateExportHeader)
include(group_files_recursive)

find_package(OpenMP)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(DATA_SETS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data_sets")
set(OUT_PATH "${CMAKE_BINARY_DIR}/bin")

file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(machine_learning_methods ${Boost_LIBRARIES})
