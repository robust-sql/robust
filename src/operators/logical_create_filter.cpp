#include "duckdb/main/database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "logical_create_filter.hpp"
#include "logical_probe_filter.hpp"
#include "physical_create_filter.hpp"
#include "physical_probe_filter.hpp"
#include <utility>

namespace duckdb {

LogicalCreateFilter::LogicalCreateFilter() : LogicalExtensionOperator() {
	this->type = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
	message = "CREATE_FILTER";
}

LogicalCreateFilter::LogicalCreateFilter(const FilterOperation &filter_op)
    : LogicalExtensionOperator(), filter_operation(filter_op), input_bindings(filter_op.build_columns) {
	this->type = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
	message = "CREATE_FILTER";
}

InsertionOrderPreservingMap<string> LogicalCreateFilter::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Operator"] = "LogicalCreateFilter";
	result["Build Table"] = to_string(filter_operation.build_table_idx);
	// there can be multiple probe tables for a single create
	string probe_tables;
	vector<idx_t> seen_probe;
	for (const auto &col : filter_operation.probe_columns) {
		bool found = false;
		for (auto idx : seen_probe) {
			if (idx == col.table_index) {
				found = true;
				break;
			}
		}
		if (!found) {
			if (!probe_tables.empty()) {
				probe_tables += ", ";
			}
			probe_tables += to_string(col.table_index);
			seen_probe.push_back(col.table_index);
		}
	}
	result["Probe Tables"] = probe_tables;

	string build_cols = "";
	for (size_t i = 0; i < filter_operation.build_columns.size(); i++) {
		if (i > 0) {
			build_cols += ", ";
		}
		build_cols += "(" + to_string(filter_operation.build_columns[i].table_index) + "." +
		              to_string(filter_operation.build_columns[i].column_index) + ")";
	}
	result["Build Columns"] = build_cols;

	if (estimated_cardinality != DConstants::INVALID_INDEX) {
		result["Estimated Cardinality"] = std::to_string(estimated_cardinality);
	}

	return result;
}

vector<ColumnBinding> LogicalCreateFilter::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

void LogicalCreateFilter::ResolveTypes() {
	if (!children.empty() && children[0]) {
		types = children[0]->types;
	}
}

PhysicalOperator &LogicalCreateFilter::CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) {
	if (!physical) {
		// step 1: get child column bindings to understand chunk schema
		vector<ColumnBinding> child_bindings = children[0]->GetColumnBindings();

		// step 2: resolve/map the filter operation columns to chunk column indices.
		// resolved_indices stores the columns on which the bloom filters are
		// built.
		// TODO: optimize: Use a map for filter_operation.build_columns to speed up lookup
		vector<idx_t> resolved_indices;
		for (const ColumnBinding &column_binding : input_bindings) {
			// find the position of the filter column ColumnBinding in the chunk columns
			for (idx_t i = 0; i < child_bindings.size(); i++) {
				if (child_bindings[i].table_index == column_binding.table_index &&
				    child_bindings[i].column_index == column_binding.column_index) {
					resolved_indices.push_back(i);
					break;
				}
			}
		}

		// step 3: create physical operator with the resolved indices
		PhysicalOperator &physical_op = generator.Make<PhysicalCreateFilter>(
		    make_shared_ptr<FilterOperation>(filter_operation), types, estimated_cardinality, resolved_indices);
		// auto filter_plan = FilterOperationToFilterPlan(filter_operation);
		// auto &physical_op = generator.Make<PhysicalCreateFilter>(make_shared<FilterOperation>(filter_operation),
		// types, estimated_cardinality);
		for (auto &child : children) {
			auto &child_physical = generator.CreatePlan(*child);
			physical_op.children.emplace_back(child_physical);
		}
		physical = static_cast<PhysicalCreateFilter *>(&physical_op);

		// propagate dynamic filter pushdown targets
		for (auto &target : pushdown_targets) {
			PhysicalCreateFilter::DynamicFilterTarget phys_target;
			phys_target.dynamic_filters = target.dynamic_filters;
			phys_target.scan_column_index = target.scan_column_index;
			phys_target.probe_column = target.probe_column;
			phys_target.column_type = target.column_type;
			phys_target.column_name = target.column_name;
			physical->pushdown_targets.push_back(std::move(phys_target));
		}
		physical->is_forward_pass = is_forward_pass;

		// link back to related PROBE_FILTER operators
		// the links are used to create pipeline dependencies
		for (const LogicalProbeFilter *probe_filter : related_probe_filter) {
			if (probe_filter->physical) {
				// TODO: keep either related_create_filter or related_create_filter_vec. Not both. Most likely we'll
				// have to remove related_create_filter.
				probe_filter->physical->related_create_filter = physical;
				probe_filter->physical->related_create_filter_vec.push_back(physical);
			}
		}
		return physical_op;
	}
	return *physical;
}
} // namespace duckdb
