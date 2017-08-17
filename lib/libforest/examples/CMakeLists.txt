cmake_minimum_required(VERSION 2.6)

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Curses REQUIRED)
include_directories(../include ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CURSES_INCLUDE_PATH})

#add_executable(cli_convert convert.cpp)
#target_link_libraries(cli_convert libforest ${Boost_LIBRARIES})

add_executable(cli_rf rf.cpp)
target_link_libraries(cli_rf libforest ${Boost_LIBRARIES})

#add_executable(cli_adaboost adaboost.cpp)
#target_link_libraries(cli_adaboost libforest ${Boost_LIBRARIES})

add_executable(cli_decision_tree decision_tree.cpp)
target_link_libraries(cli_decision_tree libforest ${Boost_LIBRARIES})

#add_executable(cli_pixel_importance pixel_importance.cpp)
#target_link_libraries(cli_pixel_importance libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(cli_online_decision_tree online_decision_tree.cpp)
#target_link_libraries(cli_online_decision_tree libforest ${Boost_LIBRARIES})

#add_executable(cli_online_rf online_rf.cpp)
#target_link_libraries(cli_online_rf libforest ${Boost_LIBRARIES})

#add_executable(cli_online_rf_convergence online_rf_convergence.cpp)
#target_link_libraries(cli_online_rf_convergence libforest ${Boost_LIBRARIES})

#add_executable(cli_density_tree density_tree.cpp)
#target_link_libraries(cli_density_tree libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(cli_density_forest density_forest.cpp)
#target_link_libraries(cli_density_forest libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(cli_kde kde.cpp)
#target_link_libraries(cli_kde libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(cli_kernel_density_tree kernel_density_tree.cpp)
#target_link_libraries(cli_kernel_density_tree libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(cli_kmeans kmeans.cpp)
target_link_libraries(cli_kmeans libforest ${Boost_LIBRARIES})
