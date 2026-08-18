cmake_minimum_required(VERSION 3.28)

project(CMakeSFMLProject LANGUAGES CXX)

# ============================================================
# Project settings
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin
)

# ============================================================
# OpenMP
# ============================================================

find_package(OpenMP REQUIRED)

# ============================================================
# SFML
# ============================================================

include(FetchContent)

FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)

FetchContent_MakeAvailable(SFML)

# ============================================================
# Executable
# ============================================================

add_executable(
    sim
    src/main.cpp
)

# ============================================================
# Compiler features
# ============================================================

target_compile_features(
    sim
    PRIVATE
        cxx_std_20
)

# ============================================================
# Compile options
# ============================================================

target_compile_options(
    sim
    PRIVATE
        $<$<CONFIG:Release>:-O3>
)

# ============================================================
# Libraries
# ============================================================

target_link_libraries(
    sim
    PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
        OpenMP::OpenMP_CXX
)

# ============================================================
# IDE settings
# ============================================================

set_property(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT sim
)
