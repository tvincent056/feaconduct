cmake_minimum_required (VERSION 3.1)
project (featool)
set (CMAKE_CXX_STANDARD 11)
find_package(MPI REQUIRED)
set(LIBMESH_LIBRARY_PATH /usr/local/lib CACHE PATH "path to libmesh library directory")
set(LIBMESH_INCLUDE_PATH /usr/local/include CACHE PATH "path to libmesh include directory")
set(EIGEN_INCLUDE_PATH /usr/local/include CACHE PATH "path to eigen include directory")
add_library(libmesh SHARED IMPORTED)
set_target_properties(libmesh PROPERTIES
  IMPORTED_LOCATION "${LIBMESH_LIBRARY_PATH}/libmesh_opt.so"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBMESH_INCLUDE_PATH};${EIGEN_INCLUDE_PATH}"
)
set(PETSC_INCLUDE_PATH /usr/local/include CACHE PATH "path to petsc include directory")
set(PETSC_LIBRARY_PATH /usr/local/lib CACHE PATH "path to petsc library directory")
add_library(petsc SHARED IMPORTED)
set_target_properties(petsc PROPERTIES
  IMPORTED_LOCATION "${PETSC_LIBRARY_PATH}/libpetsc.so"
  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_PATH}"
)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_executable(featool "")
add_subdirectory(src)
target_link_libraries(featool libmesh)
target_link_libraries(featool ${MPI_LIBRARIES};${MPI_C_LIBRARIES})
target_link_libraries(featool petsc -pthread -lm)
