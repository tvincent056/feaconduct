cmake_minimum_required (VERSION 3.1)
project (feaconduct)
set (CMAKE_CXX_STANDARD 11)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED)
include(ExternalProject)

if (NOT LIBMESH_SRC_PATH)
  message(SEND_ERROR "must define LIBMESH_SRC_PATH")
endif(NOT LIBMESH_SRC_PATH)
if (NOT EIGEN_INCLUDE_PATH)
  message(SEND_ERROR "must define EIGEN_INCLUDE_PATH")
endif(NOT EIGEN_INCLUDE_PATH)
if (NOT PETSC_SRC_PATH)
  message(SEND_ERROR "must define PETSC_SRC_PATH")
endif(NOT PETSC_SRC_PATH)

string( TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)

set(PETSC_ARCH ${CMAKE_BUILD_TYPE})
set(PETSC_BIN ${PETSC_SRC_PATH}/${PETSC_ARCH})
set(PETSC_INSTALL_DIR ${PROJECT_BINARY_DIR}/petsc)
set(PETSC_INCLUDE_PATH ${PETSC_INSTALL_DIR}/include)
set(PETSC_LIB_DIR ${PETSC_INSTALL_DIR}/lib)
set(PETSC_LIBRARIES ${PETSC_LIB_DIR}/libpetsc.so)
set(PETSC_CONFIG_OPTIONS --with-mpi-dir=${MPI_INCLUDE_PATH}/.. -PETSC_ARCH=${PETSC_ARCH})
if (build_type_lower STREQUAL "release")
  set(PETSC_CONFIG_OPTIONS ${PETSC_CONFIG_OPTIONS} --with-debugging=0)
endif()
ExternalProject_Add(
  petsc
  PREFIX ${PETSC_SRC_PATH}
  SOURCE_DIR ${PETSC_SRC_PATH}
  CONFIGURE_COMMAND ./configure --prefix=${PETSC_INSTALL_DIR} ${PETSC_CONFIG_OPTIONS}
  BUILD_COMMAND make
  BINARY_DIR ${PETSC_SRC_PATH}
)

#add_library(petsc SHARED IMPORTED)
#set_target_properties(petsc PROPERTIES
#  IMPORTED_LOCATION "${PROJECT_BINARY_DIR}/petsc/lib/libpetsc.so"
#  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_PATH}"
#)

if ( build_type_lower STREQUAL "debug")
  set(LIBMESH_BUILD_TYPE "dbg")
else ()
  set(LIBMESH_BUILD_TYPE "opt")
endif()
set(LIBMESH_INSTALL_DIR ${PROJECT_BINARY_DIR}/libmesh)
set(LIBMESH_INCLUDE_PATH ${LIBMESH_INSTALL_DIR}/include)
set(LIBMESH_LIB_DIR ${LIBMESH_INSTALL_DIR}/lib)
set(LIBMESH_LIBRARIES ${LIBMESH_LIB_DIR}/libmesh_${LIBMESH_BUILD_TYPE}.so)
set(LIBMESH_CONFIG_OPTIONS --prefix=${LIBMESH_INSTALL_DIR} --enable-static --enable-petsc-required --with-mpi=${MPI_INCLUDE_PATH}/.. --with-boost=${Boost_INCLUDE_DIR}/.. --with-methods=${LIBMESH_BUILD_TYPE})

ExternalProject_Add(
  libMesh
  DEPENDS petsc
  PREFIX ${LIBMESH_SRC_PATH}
  SOURCE_DIR ${LIBMESH_SRC_PATH}
  CONFIGURE_COMMAND PETSC_DIR=${PETSC_INSTALL_DIR} ./configure ${LIBMESH_CONFIG_OPTIONS}
  BINARY_DIR ${LIBMESH_SRC_PATH}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
)

#add_library(libmesh SHARED IMPORTED)
#set_target_properties(libmesh PROPERTIES
#  IMPORTED_LOCATION ${LIBMESH_LIB_DIR}/libmesh_opt.so
#  INTERFACE_INCLUDE_DIRECTORIES "${LIBMESH_INCLUDE_DIR};${EIGEN_INCLUDE_PATH}"
#)

#set(PETSC_SRC_PATH /usr/local CACHE PATH "path to petsc include directory")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${LIBMESH_INCLUDE_PATH})
include_directories(SYSTEM ${PETSC_INCLUDE_PATH})
add_executable(feaconduct "")
add_subdirectory(src)
target_link_libraries(feaconduct ${LIBMESH_LIBRARIES})
target_link_libraries(feaconduct ${PETSC_LIBRARIES} -pthread -lm)
target_link_libraries(feaconduct ${MPI_LIBRARIES};${MPI_C_LIBRARIES})
