cmake_minimum_required(VERSION 3.28)
project(CMakeSFMLProject LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 20)

add_executable(sim src/main.cpp)

# 1. Fetch SFML normally (Without OpenMP polluting it)
include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(SFML)

# 2. Link SFML to your executable
target_compile_features(sim PRIVATE cxx_std_20)
target_link_libraries(sim PRIVATE sfml-graphics sfml-window sfml-system)

# 3. Target-Specific OpenMP for Mac (Applies ONLY to 'sim')
if(APPLE)
    target_include_directories(sim PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(sim PRIVATE /opt/homebrew/opt/libomp/lib)
    target_compile_options(sim PRIVATE -Xpreprocessor -fopenmp)
    target_link_options(sim PRIVATE -Xpreprocessor -fopenmp) # Ensure linker knows about threads
    target_link_libraries(sim PRIVATE omp)
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT main)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Xpreprocessor -fopenmp")
