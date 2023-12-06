project(vipss)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")

find_package(NLopt REQUIRED)
set(NLOPT_LIB_DIR "")
set(NLOPT_LIB ${NLOPT_LIBRARIES})

find_package(Armadillo REQUIRED)
set(ARMADILLO_LIB_DIRS "")
set(ARMADILLO_LIB ${ARMADILLO_LIBRARIES})

set(Eigen_dir "./include/Eigen")

find_package( OpenCV REQUIRED )
include_directories( ${Eigen_dir} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${NLOPT_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ./vipss_src/surfacer)
# aux_source_directory(main.cpp MAIN)
aux_source_directory(./vipss_src SRC_LIST)
# include(CMakePrintHelpers)
# cmake_print_variables(SRC_LIST)
aux_source_directory(./vipss_src/surfacer SURFACER_LIST)

LINK_DIRECTORIES(${ARMADILLO_LIB_DIRS} ${NLOPT_LIB_DIR})
add_library(rbf ${SRC_LIST})
add_library(surface ${SURFACER_LIST})
add_executable(${PROJECT_NAME} main.cpp )

target_link_libraries(${PROJECT_NAME} rbf surface ${ARMADILLO_LIB} ${NLOPT_LIB})


file(GLOB SOURCE_FILES
     "src/*.h"
     "src/*.cpp"
)

add_library(ImgPro ${SOURCE_FILES})
target_link_libraries(ImgPro ${OpenCV_LIBS})

add_executable(img_pro main0.cpp)
target_link_libraries(img_pro ImgPro ${OpenCV_LIBS} stdc++fs)

add_executable(test_fun test.cpp )
target_link_libraries(test_fun rbf surface ${ARMADILLO_LIB} ${NLOPT_LIB})
