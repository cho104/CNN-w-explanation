cmake_minimum_required(VERSION 3.14)
project(ConvNetVisualizer)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically download and compile Raylib
include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG master
)
FetchContent_MakeAvailable(raylib)

# Compile your main file
add_executable(${PROJECT_NAME} main.cpp)

# Link Raylib to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

# Make sure the executable can find your .txt files by copying them to the build folder
file(GLOB TEXT_FILES "*.txt")
foreach(TXT_FILE ${TEXT_FILES})
    configure_file(${TXT_FILE} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()