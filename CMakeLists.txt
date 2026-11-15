cmake_minimum_required(VERSION 3.28)
project(FastMultipoleMethod LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(SFML)

add_executable(sim src/main.cpp)
target_compile_features(sim PRIVATE cxx_std_20)
target_link_libraries(sim PRIVATE SFML::Graphics)


if(MSVC)
    target_compile_options(sim PRIVATE /openmp:llvm)
else()
    if(APPLE AND NOT OpenMP_ROOT)
        find_program(BREW brew)
        if(BREW)
            execute_process(
                COMMAND ${BREW} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(LIBOMP_PREFIX)
                set(OpenMP_ROOT ${LIBOMP_PREFIX})
            endif()
        endif()
    endif()
    find_package(OpenMP REQUIRED)
    target_link_libraries(sim PRIVATE OpenMP::OpenMP_CXX)
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT sim)