//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_create_filter.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "../optimizer/graph_manager.hpp"

namespace duckdb {
class DatabaseInstance;
class PhysicalCreateFilter;
class LogicalProbeFilter;

class LogicalCreateFilter : public LogicalExtensionOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
	static constexpr auto OPERATOR_TYPE_NAME = "logical_create_filter";

public:
	explicit LogicalCreateFilter();
	explicit LogicalCreateFilter(const FilterOperation &filter_op);

	bool can_stop = false;
	FilterOperation filter_operation;
	vector<ColumnBinding> input_bindings;
	PhysicalCreateFilter *physical = nullptr;

	vector<LogicalProbeFilter *> related_probe_filter;
	bool is_forward_pass = false;

	struct DynamicFilterTarget {
		shared_ptr<DynamicTableFilterSet> dynamic_filters;
		idx_t scan_column_index;
		ColumnBinding probe_column;
		LogicalType column_type;
		string column_name;
	};
	vector<DynamicFilterTarget> pushdown_targets;

	string message;

public:
	string GetExtensionName() const override {
		return "rpt";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override;
	vector<ColumnBinding> GetColumnBindings() override;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) override;

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
