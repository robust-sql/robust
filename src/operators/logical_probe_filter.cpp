#include "logical_probe_filter.hpp"
#include "physical_probe_filter.hpp"
#include "utils/debug_utils.hpp"

namespace duckdb {

LogicalProbeFilter::LogicalProbeFilter() : LogicalExtensionOperator() {
	this->type = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
}

LogicalProbeFilter::LogicalProbeFilter(const FilterOperation &filter_op)
    : LogicalExtensionOperator(), filter_operation(filter_op), input_bindings(filter_op.probe_columns) {
	this->type = LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
}

InsertionOrderPreservingMap<string> LogicalProbeFilter::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Operator"] = "LogicalProbeFilter";

	result["Build Table"] = to_string(filter_operation.build_table_idx);
	result["Probe Table"] = to_string(filter_operation.probe_table_idx);

	string probe_cols = "";
	for (size_t i = 0; i < filter_operation.probe_columns.size(); i++) {
		if (i > 0) {
			probe_cols += ", ";
		}
		probe_cols += "(" + to_string(filter_operation.probe_columns[i].table_index) + "." +
		              to_string(filter_operation.probe_columns[i].column_index) + ")";
	}
	result["Probe Columns"] = probe_cols;

	if (estimated_cardinality != DConstants::INVALID_INDEX) {
		result["Estimated Cardinality"] = std::to_string(estimated_cardinality);
	}

	return result;
}

vector<ColumnBinding> LogicalProbeFilter::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

void LogicalProbeFilter::ResolveTypes() {
	if (!children.empty() && children[0]) {
		types = children[0]->types;
	}
}

PhysicalOperator &LogicalProbeFilter::CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) {
	if (!physical) {
		// step 1: get child column bindings to understand chunk schema
		vector<ColumnBinding> child_bindings = children[0]->GetColumnBindings();

		// step 2: resolve/map the filter operation probe columns to chunk column indices
		vector<idx_t> resolved_indices;

#ifdef DEBUG
		Printer::Print(StringUtil::Format("[RESOLVE] LogicalProbeFilter probe_table=%llu has %zu probe_columns",
		                                  (unsigned long long)filter_operation.probe_table_idx,
		                                  filter_operation.probe_columns.size()));
		Printer::Print(StringUtil::Format("[RESOLVE] child_bindings.size()=%zu", child_bindings.size()));
		for (idx_t j = 0; j < child_bindings.size(); j++) {
			Printer::Print(StringUtil::Format("  child_bindings[%llu] = table_idx=%llu, col_idx=%llu",
			                                  (unsigned long long)j, (unsigned long long)child_bindings[j].table_index,
			                                  (unsigned long long)child_bindings[j].column_index));
		}
#endif

		for (const ColumnBinding &column_binding : input_bindings) {
			D_PRINTF("[RESOLVE] Looking for probe_column: table_idx=%llu, col_idx=%llu",
			         (unsigned long long)column_binding.table_index, (unsigned long long)column_binding.column_index);
			// find the position of the filter column ColumnBinding in the chunk columns
			for (idx_t i = 0; i < child_bindings.size(); i++) {
				if (child_bindings[i].table_index == column_binding.table_index &&
				    child_bindings[i].column_index == column_binding.column_index) {
					resolved_indices.push_back(i);
					D_PRINTF("[RESOLVE] Matched at chunk position %llu", (unsigned long long)i);
					break;
				}
			}
		}

		// step 3: create physical operator with the resolved indices
		auto &plan = generator.CreatePlan(*children[0]);
		PhysicalOperator &physical_op = generator.Make<PhysicalProbeFilter>(
		    make_shared_ptr<FilterOperation>(filter_operation), plan.types, estimated_cardinality, resolved_indices);
		physical = static_cast<PhysicalProbeFilter *>(&physical_op);
		physical->is_passthrough = is_passthrough;

		// set up reference to related PhysicalCreateFilter if available
		if (related_create_filter) {
			D_PRINTF("[LOGICAL USE] probe table - table_%llu Related_create_filter exists",
			         (unsigned long long)filter_operation.probe_table_idx);
		}
		if (related_create_filter && related_create_filter->physical) {
			D_PRINTF("[LOGICAL USE] probe table - table_%llu Related_create_filter physical exists",
			         (unsigned long long)filter_operation.probe_table_idx);
			physical->related_create_filter = related_create_filter->physical;
			physical->related_create_filter_vec.push_back(related_create_filter->physical);
		}

		physical_op.children.emplace_back(plan);
		return physical_op;
	}
	return *physical;
}

} // namespace duckdb
