# - Find Google Test
# Find the native GTest includes and library
#
#  GTest_INCLUDES    - where to find gtest.h
#  GTest_LIBRARIES   - List of libraries when using GTest.
#  GTest_FOUND       - True if GTest found.

find_path (GTest_INCLUDE_DIR gtest/
    PATHS ${GTest_ROOT_DIR} ~/gtest-1.7.0/include/)
find_library (GTest_LIBRARIES NAMES gtest libgtest gtest_main libgtest_main
    PATHS ${GTest_ROOT_DIR} ~/gtest-1.7.0/)

mark_as_advanced (GTest_INCLUDE_DIR GTest_LIBRARIES)
