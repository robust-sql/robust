PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=robust
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
OVERRIDE_GIT_DESCRIBE ?= v2.0.0

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Vendor Robust-owned benchmark suites + duckdb patches into the submodule.
# Runs automatically before release/debug builds. Safe to re-run; idempotent.
# See scripts/vendor_duckdb_bench.sh for details.
.PHONY: vendor-duckdb-bench
vendor-duckdb-bench:
	@bash $(PROJ_DIR)scripts/vendor_duckdb_bench.sh

release: vendor-duckdb-bench
debug: vendor-duckdb-bench
