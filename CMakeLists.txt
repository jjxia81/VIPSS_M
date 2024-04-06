cmake_minimum_required(VERSION 3.6)
project(vipss_m)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 ")

set(CMAKE_IGNORE_PREFIX_PATH $ENV{CONDA_PREFIX})
message(STATUS "$ENV{CONDA_PREFIX}")


find_package(OpenMP REQUIRED)

# find_package(NLopt REQUIRED)
# set(NLOPT_LIB_DIR "")
# set(NLOPT_LIB ${NLOPT_LIBRARIES})

find_package(Armadillo REQUIRED)
set(ARMADILLO_LIB_DIRS "")
set(ARMADILLO_LIB ${ARMADILLO_LIBRARIES})

# set(Eigen_dir "./include/Eigen")

find_package (Eigen3 REQUIRED)

find_package(yaml-cpp REQUIRED)
include_directories( ${YAML_CPP_INCLUDE_DIRS} )

find_package(NLopt REQUIRED)
set(NLOPT_LIB_DIR "")
set(NLOPT_LIB ${NLOPT_LIBRARIES})

find_package(LAPACK)
find_package(OpenMP)
find_package(SPQR)
find_package(CHOLMOD)
find_package(UMFPACK)
find_package(Open3D REQUIRED)


message(SPQR_LIBRARY_PATH=${SPQR_LIBRARIES})
message(CHOLMOD_LIBRARY_PATH=${CHOLMOD_LIBRARIES})
message(PASTIX_INCLUDE_DIR=${PASTIX_DIR})
# find_package (SuiteSparseBLAS )

find_package( OpenCV REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories(${SPQR_INCLUDE_DIR})
include_directories(/usr/local/include)
# include_directories(/usr/local/include/pastix)
include_directories(${Open3D_INCLUDE_DIR})

# include_directories( ${SuiteSparseBLAS_INCLUDE_DIR} )

# include_directories( /usr/local/include/suitesparse )

# message(SuiteSparseBLAS_INCLUDE_DIR=${SuiteSparseBLAS_INCLUDE_DIR})

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${NLOPT_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ./vipss_src ./vipss_src/surfacer)
# aux_source_directory(main.cpp MAIN)
aux_source_directory(./vipss_src SRC_LIST)
# include(CMakePrintHelpers)
# cmake_print_variables(SRC_LIST)
aux_source_directory(./vipss_src/surfacer SURFACER_LIST)

LINK_DIRECTORIES(${ARMADILLO_LIB_DIRS} ${NLOPT_LIB_DIR})
add_library(rbf ${SRC_LIST})
target_link_libraries(rbf ${YAML_CPP_LIBRARIES} OpenMP::OpenMP_CXX spqr)
add_library(surface ${SURFACER_LIST})
# add_executable(${PROJECT_NAME} main.cpp )

# target_link_libraries(${PROJECT_NAME} rbf surface ${ARMADILLO_LIB} ${NLOPT_LIB} ${YAML_CPP_LIBRARIES} OpenMP::OpenMP_CXX)

file(GLOB SOURCE_FILES
     "src/*.h"
     "src/*.hpp"
     "src/*.cpp"
)



add_library(ImgPro ${SOURCE_FILES})
target_link_libraries(ImgPro ${OpenCV_LIBS})

add_executable(img_pro main0.cpp)
target_link_libraries(img_pro ImgPro ${OpenCV_LIBS} stdc++fs)

add_executable(${PROJECT_NAME} main.cpp )
target_link_libraries(${PROJECT_NAME} rbf surface ${ARMADILLO_LIB}  ${YAML_CPP_LIBRARIES} 
                         OpenMP::OpenMP_CXX ${NLOPT_LIBRARIES} ${LAPACK_LIBRARIES} 
                         ${SPQR_LIBRARIES} ${CHOLMOD_LIBRARIES} ${UMFPACK_LIBRARIES} ${Open3D_LIBRARIES} )


# find_package(CGAL REQUIRED)
# include_directories( ${CGAL_dir} )
# message(STATUS ${CGAL_dir})
# add_executable(normal_estimator normal_estimation.cpp)
# target_link_libraries(normal_estimator stdc++fs )

# add_executable(mesh_slicer mesh_slicer.cpp)
# target_link_libraries(mesh_slicer stdc++fs )