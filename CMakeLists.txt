cmake_minimum_required(VERSION 3.28)
project(CMakeSFMLProject LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)

add_executable(sim src/main.cpp)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(SFML)

target_compile_features(sim PRIVATE cxx_std_20)
target_link_libraries(sim PRIVATE sfml-graphics sfml-window sfml-system)

find_package(OpenMP REQUIRED)
target_link_libraries(sim PRIVATE OpenMP::OpenMP_CXX)

# Keep only if needed on Apple
if(APPLE)
    target_include_directories(sim PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(sim PRIVATE /opt/homebrew/opt/libomp/lib)
endif()

# Better than overwriting global CMAKE_CXX_FLAGS_RELEASE
target_compile_options(sim PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

# Optional: enable IPO/LTO in Release
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT sim)
