cmake_minimum_required(VERSION 2.6)

################################################################################
### PROJECT SETTINGS
################################################################################

project(libforest)

#### FIX THIS ##########
set(EIGEN3_INCLUDE_DIR "/home/toby/libs/eigen-3.2.7")
#### /FIX THIS ##########
set(LIBF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

################################################################################
### CMAKE OPTIONS
################################################################################

# Let's you switch assertions on/off
# This must be switched on for the unit tests
OPTION(ENABLE_ASSERT "Enables assertions throughout the library" On)
IF(ENABLE_ASSERT)
    ADD_DEFINITIONS(-DLIBF_ENABLE_ASSERT)
ENDIF(ENABLE_ASSERT)

OPTION(BUILD_TESTS "If on, gtest is built and the unit tests are built" Off)

################################################################################
### THIRD PARTY LIBRARIES
################################################################################

find_package(Boost COMPONENTS system REQUIRED)
#find_package(Eigen3 REQUIRED)
#### FIX THIS ##########
include_directories(/home/toby/libs/eigen-3.2.7)
#### /FIX THIS ##########
find_package(Curses REQUIRED)
# TODO: For PixelImportanceTool, should not be required - should be optional
# via CMake option!
# TODO: Also for OpenCV matrix reader. Add general "OpenCV Package option"
find_package(OpenCV REQUIRED) 
# TODO: Add general MATLAB option for importing MATLAB mat files
find_package(OpenMP)

################################################################################
### SET UP BUILD ENVIRONMENT
################################################################################

# Add the include directories of the third party libraries
include_directories(
                    ${LIBF_INCLUDE_DIRS} 
                    ${Boost_INCLUDE_DIR} 
                    ${OpenCV_INCLUDE_DIRS} 
                    lib/gtest-1.7.0/include
                    ${CURSES_INCLUDE_PATH})

# Add the necessary g++ flags:
# -std=c++11: C++11
# -g: Debuggin with Valgrind
# -Wall: All warnings
# OpenMP
set(CMAKE_CXX_FLAGS "-g -Wall ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
################################################################################
### SET UP TARGETS
################################################################################

# Build the actual library
add_library(libforest STATIC 
                    src/classifier.cpp 
                    src/classifier_learning.cpp 
                    src/classifier_learning_tools.cpp 
                    src/classifier_tools.cpp 
                    src/data.cpp 
                    src/data_tools.cpp
                    src/error_handling.cpp
                    src/estimators.cpp
                    src/learning_tools.cpp
                    src/unsupervised_learning.cpp
                    src/util.cpp)

target_link_libraries(libforest 
                    ${OpenCV_LIBRARIE}
                    ${OpenCV_LIBS}
                    ${CURSES_CURSES_LIBRARY}
                    ${CURSES_NCURSES_LIBRARY})

# Build the examples
# TODO: Make this a CMake option or remove it completely from the build process
#add_subdirectory(examples)

################################################################################
### SET UP UNIT TESTS
################################################################################


