# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo - build as dynamic extension 
duckdb_extension_load(robust
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src
    LOAD_TESTS
)

# Any extra extensions that should be built
# e.g.: duckdb_extension_load(json)

# tpch: statically linked so the benchmark_runner can CALL dbgen / run the
# tpch benchmark suites (scripts/bench_tpch.sh). Required by `require tpch`.
duckdb_extension_load(tpch)
duckdb_extension_statically_link(tpch)
