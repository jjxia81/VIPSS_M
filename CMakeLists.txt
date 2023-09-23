project(vipss)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")

find_package(nlopt REQUIRED)
set(NLOPT_LIB_DIR "")
set(NLOPT_LIB ${NLOPT_LIBRARIES})

find_package(Armadillo REQUIRED)
set(ARMADILLO_LIB_DIRS "")
set(ARMADILLO_LIB ${ARMADILLO_LIBRARIES})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${NLOPT_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ./vipss_src/surfacer)
# aux_source_directory(main.cpp MAIN)
aux_source_directory(./vipss_src SRC_LIST)
aux_source_directory(./vipss_src/surfacer SURFACER_LIST)

LINK_DIRECTORIES(${ARMADILLO_LIB_DIRS} ${NLOPT_LIB_DIR})
add_executable(${PROJECT_NAME} ${SRC_LIST} main.cpp ${SURFACER_LIST})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIB} ${NLOPT_LIB})


file(GLOB SOURCE_FILES
     "src/*.h"
     "src/*.cpp"
)

add_library(ImgPro ${SOURCE_FILES})
target_link_libraries(ImgPro ${OpenCV_LIBS})

add_executable(img_pro main0.cpp)
target_link_libraries(img_pro ImgPro ${OpenCV_LIBS} stdc++fs)
