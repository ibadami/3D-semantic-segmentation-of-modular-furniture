
#-------------------
# Test
#-------------------
set(PROJECT_TEST_NAME ${PRJ_NAME}_test)
include_directories(${Parser_INCLUDE_DIRS})
include_directories(BEFORE ${gtest_SOURCE_DIR}/include)

file(GLOB TEST_SRC_FILES "*.cpp")
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
target_link_libraries(${PROJECT_TEST_NAME} gtest gtest_main ${OpenCV_LIBS} parser)
add_test(test1 ${PROJECT_TEST_NAME})
