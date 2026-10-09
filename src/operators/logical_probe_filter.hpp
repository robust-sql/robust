//===----------------------------------------------------------------------===//
//                         DuckDB
//
// operator/logical_probe_filter.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "logical_create_filter.hpp"
#include "../optimizer/graph_manager.hpp"

namespace duckdb {
class DatabaseInstance;
class PhysicalProbeFilter;

class LogicalProbeFilter final : public LogicalExtensionOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
	static constexpr auto OPERATOR_TYPE_NAME = "logical_probe_filter";

public:
	explicit LogicalProbeFilter();
	explicit LogicalProbeFilter(const FilterOperation &filter_op);

	FilterOperation filter_operation;
	LogicalCreateFilter *related_create_filter = nullptr;
	bool is_passthrough = false;
	vector<ColumnBinding> input_bindings;

	PhysicalProbeFilter *physical = nullptr;

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

// void RegisterLogicalProbeFilterOperatorExtension(DatabaseInstance &instance);

} // namespace duckdb
