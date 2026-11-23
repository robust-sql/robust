#pragma once

#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/planner/column_binding.hpp"

namespace duckdb {

struct ScanFilterTarget {
	shared_ptr<DynamicTableFilterSet> dynamic_filters;
	ProjectionIndex scan_column_index;
	ColumnBinding build_column;
	LogicalType column_type;
	string column_name;
	idx_t max_prf_build_rows = 0;
};

} // namespace duckdb
