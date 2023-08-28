cmake_minimum_required(VERSION 3.2)
project(VIPSS_M)


# find_package(OpenCV REQUIRED)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# include_directories(/usr/local/include)

# add_subdirectory(lib/googletest)

# include_directories(${DIVISION_HEADERS_DIR})
# include_directories(lib/googletest/googletest/include)

# set(SOURCE_FILES src/img_gradients.cpp)
file(GLOB SOURCE_FILES
     "src/*.h"
     "src/*.cpp"
)

add_library(ImgPro ${SOURCE_FILES})
target_link_libraries(ImgPro ${OpenCV_LIBS})

add_executable(VIPSS main.cpp)
target_link_libraries(VIPSS ImgPro)
# install(TARGETS divider_tests DESTINATION bin)