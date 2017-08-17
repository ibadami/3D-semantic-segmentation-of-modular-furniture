# Install script for directory: /home/tom/Downloads/eigen/unsupported/Eigen

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/tom/Downloads/eigen/unsupported/Eigen/AdolcForward"
    "/home/tom/Downloads/eigen/unsupported/Eigen/AlignedVector3"
    "/home/tom/Downloads/eigen/unsupported/Eigen/ArpackSupport"
    "/home/tom/Downloads/eigen/unsupported/Eigen/AutoDiff"
    "/home/tom/Downloads/eigen/unsupported/Eigen/BVH"
    "/home/tom/Downloads/eigen/unsupported/Eigen/FFT"
    "/home/tom/Downloads/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/tom/Downloads/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/tom/Downloads/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/tom/Downloads/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/tom/Downloads/eigen/unsupported/Eigen/MoreVectorization"
    "/home/tom/Downloads/eigen/unsupported/Eigen/MPRealSupport"
    "/home/tom/Downloads/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/tom/Downloads/eigen/unsupported/Eigen/NumericalDiff"
    "/home/tom/Downloads/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/tom/Downloads/eigen/unsupported/Eigen/Polynomials"
    "/home/tom/Downloads/eigen/unsupported/Eigen/Skyline"
    "/home/tom/Downloads/eigen/unsupported/Eigen/SparseExtra"
    "/home/tom/Downloads/eigen/unsupported/Eigen/Splines"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/tom/Downloads/eigen/build/unsupported/Eigen/src/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

