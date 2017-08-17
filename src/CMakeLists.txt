#include directories helps to include header files
include_directories(".")

include_directories($ENV{GUROBI_HOME}/include)
link_directories(/home/badami/gurobi650/linux64/lib)
 
#get the source filenames to compile
file(GLOB_RECURSE SOURCES *.cpp)

add_definitions("${CFLAGS}")
include_directories(${Parser_INCLUDE_DIRS})
#creating libraries 
add_library(${PRJ_NAME} STATIC ${SOURCES})

