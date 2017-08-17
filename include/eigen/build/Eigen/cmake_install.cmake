# Install script for directory: /home/tom/Downloads/eigen/Eigen

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/tom/Downloads/eigen/Eigen/CholmodSupport"
    "/home/tom/Downloads/eigen/Eigen/Core"
    "/home/tom/Downloads/eigen/Eigen/SparseLU"
    "/home/tom/Downloads/eigen/Eigen/LU"
    "/home/tom/Downloads/eigen/Eigen/StdList"
    "/home/tom/Downloads/eigen/Eigen/Eigen"
    "/home/tom/Downloads/eigen/Eigen/StdVector"
    "/home/tom/Downloads/eigen/Eigen/Sparse"
    "/home/tom/Downloads/eigen/Eigen/PaStiXSupport"
    "/home/tom/Downloads/eigen/Eigen/LeastSquares"
    "/home/tom/Downloads/eigen/Eigen/PardisoSupport"
    "/home/tom/Downloads/eigen/Eigen/SparseCore"
    "/home/tom/Downloads/eigen/Eigen/SparseCholesky"
    "/home/tom/Downloads/eigen/Eigen/OrderingMethods"
    "/home/tom/Downloads/eigen/Eigen/Eigenvalues"
    "/home/tom/Downloads/eigen/Eigen/Eigen2Support"
    "/home/tom/Downloads/eigen/Eigen/Householder"
    "/home/tom/Downloads/eigen/Eigen/UmfPackSupport"
    "/home/tom/Downloads/eigen/Eigen/SuperLUSupport"
    "/home/tom/Downloads/eigen/Eigen/Dense"
    "/home/tom/Downloads/eigen/Eigen/Jacobi"
    "/home/tom/Downloads/eigen/Eigen/MetisSupport"
    "/home/tom/Downloads/eigen/Eigen/StdDeque"
    "/home/tom/Downloads/eigen/Eigen/Cholesky"
    "/home/tom/Downloads/eigen/Eigen/Geometry"
    "/home/tom/Downloads/eigen/Eigen/IterativeLinearSolvers"
    "/home/tom/Downloads/eigen/Eigen/QR"
    "/home/tom/Downloads/eigen/Eigen/QtAlignedMalloc"
    "/home/tom/Downloads/eigen/Eigen/Array"
    "/home/tom/Downloads/eigen/Eigen/SVD"
    "/home/tom/Downloads/eigen/Eigen/SparseQR"
    "/home/tom/Downloads/eigen/Eigen/SPQRSupport"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/tom/Downloads/eigen/build/Eigen/src/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

