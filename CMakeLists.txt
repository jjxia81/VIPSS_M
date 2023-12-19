cmake_minimum_required(VERSION 3.6)
project(vipss)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 ")

set(CMAKE_IGNORE_PREFIX_PATH $ENV{CONDA_PREFIX})
message(STATUS "$ENV{CONDA_PREFIX}")


find_package(OpenMP REQUIRED)

find_package(NLopt REQUIRED)
set(NLOPT_LIB_DIR "")
set(NLOPT_LIB ${NLOPT_LIBRARIES})

find_package(Armadillo REQUIRED)
set(ARMADILLO_LIB_DIRS "")
set(ARMADILLO_LIB ${ARMADILLO_LIBRARIES})

set(Eigen_dir "./include/Eigen")

find_package(yaml-cpp REQUIRED)
include_directories( ${YAML_CPP_INCLUDE_DIRS} )

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
target_link_libraries(rbf ${YAML_CPP_LIBRARIES} OpenMP::OpenMP_CXX)
add_library(surface ${SURFACER_LIST})
add_executable(${PROJECT_NAME} main.cpp )

target_link_libraries(${PROJECT_NAME} rbf surface ${ARMADILLO_LIB} ${NLOPT_LIB} ${YAML_CPP_LIBRARIES} OpenMP::OpenMP_CXX)

file(GLOB SOURCE_FILES
     "src/*.h"
     "src/*.cpp"
)



add_library(ImgPro ${SOURCE_FILES})
target_link_libraries(ImgPro ${OpenCV_LIBS})

add_executable(img_pro main0.cpp)
target_link_libraries(img_pro ImgPro ${OpenCV_LIBS} stdc++fs)

add_executable(test_fun test.cpp )
target_link_libraries(test_fun rbf surface ${ARMADILLO_LIB} ${NLOPT_LIB} ${YAML_CPP_LIBRARIES} OpenMP::OpenMP_CXX)


find_package(CGAL REQUIRED)
include_directories( ${CGAL_dir} )
message(STATUS ${CGAL_dir})
add_executable(normal_estimator normal_estimation.cpp)
target_link_libraries(normal_estimator stdc++fs )