# Install script for directory: /home/tom/Downloads/eigen/Eigen/src/SparseCore

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen/src/SparseCore" TYPE FILE FILES
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseDot.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseRedux.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseSelfAdjointView.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/AmbiVector.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseCwiseUnaryOp.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseTriangularView.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparsePermutation.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/MappedSparseMatrix.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseVector.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/TriangularSolver.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseView.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseCwiseBinaryOp.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseSparseProductWithPruning.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseBlock.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseProduct.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseColEtree.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseDenseProduct.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/CompressedStorage.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/ConservativeSparseSparseProduct.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseFuzzy.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseTranspose.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseUtil.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseMatrix.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseDiagonalProduct.h"
    "/home/tom/Downloads/eigen/Eigen/src/SparseCore/SparseMatrixBase.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

