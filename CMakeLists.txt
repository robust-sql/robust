cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME robust)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src)

set(EXTENSION_SOURCES
    src/robust_extension.cpp
    src/bloom_filter.cpp
    src/operators/logical_create_filter.cpp
    src/operators/physical_create_filter.cpp
    src/operators/logical_probe_filter.cpp
    src/operators/physical_probe_filter.cpp
    src/optimizer/robust_optimizer.cpp
    src/optimizer/redundant_pair_removal.cpp
    src/optimizer/table_manager.cpp
    src/utils/dag_printer.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# Bloom Filter Benchmark (standalone executable)
option(BUILD_BLOOM_FILTER_BENCHMARK "Build the bloom filter benchmark" OFF)
if(BUILD_BLOOM_FILTER_BENCHMARK)
    add_executable(bloom_filter_benchmark
        src/benchmark/bloom_filter_benchmark.cpp
        src/bloom_filter.cpp
    )
    target_include_directories(bloom_filter_benchmark PRIVATE
        src
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(bloom_filter_benchmark duckdb)
endif()
