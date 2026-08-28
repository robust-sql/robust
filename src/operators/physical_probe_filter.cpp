#include "physical_probe_filter.hpp"
#include "physical_create_filter.hpp"
#include "bloom_filter.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "utils/debug_utils.hpp"
#include "robust_profiling.hpp"
#include "probe_empty_registry.hpp"

namespace duckdb {

PhysicalProbeFilter::PhysicalProbeFilter(PhysicalPlan &physical_plan, shared_ptr<FilterOperation> filter_operation,
                                         vector<LogicalType> types, idx_t estimated_cardinality,
                                         vector<idx_t> bound_column_indices)
    : CachingPhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      filter_operation(std::move(filter_operation)), bound_column_indices(std::move(bound_column_indices)) {
}

string PhysicalProbeFilter::GetName() const {
	return "PROBE_FILTER";
}

string PhysicalProbeFilter::ToString(ExplainFormat format) const {
	string result = "PROBE_FILTER";
	if (is_passthrough) {
		result += " (passthrough, pushed to scan)";
	} else if (filter_operation) {
		result += " [" + std::to_string(filter_operation->probe_columns.size()) + " probe columns]";
	}
	return result;
}

InsertionOrderPreservingMap<string> PhysicalProbeFilter::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Operator"] = is_passthrough ? "PhysicalProbeFilter (passthrough)" : "PhysicalProbeFilter";

	result["Build Table"] = to_string(filter_operation->build_table_idx);
	result["Probe Table"] = to_string(filter_operation->probe_table_idx);

	string probe_cols = "";
	for (size_t i = 0; i < filter_operation->probe_columns.size(); i++) {
		if (i > 0) {
			probe_cols += ", ";
		}
		probe_cols += "(" + to_string(filter_operation->probe_columns[i].table_index) + "." +
		              to_string(filter_operation->probe_columns[i].column_index) + ")";
	}
	result["Probe Columns"] = probe_cols;

	if (estimated_cardinality != DConstants::INVALID_INDEX) {
		result["Estimated Cardinality"] = std::to_string(estimated_cardinality);
	}

	return result;
}

unique_ptr<OperatorState> PhysicalProbeFilter::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<PhysicalProbeFilterState>();
}

OperatorResultType PhysicalProbeFilter::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                        GlobalOperatorState &gstate, OperatorState &state_p) const {
	// passthrough mode: filters pushed to scan, just forward data
	if (is_passthrough) {
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	if (!profiling_checked) {
		profiling_checked = true;
		auto prof = GetRobustProfilingState(context.client);
		if (prof) {
			profiling_stats =
			    prof->RegisterProbeFilter(filter_operation->build_table_idx, filter_operation->probe_table_idx,
			                              filter_operation->sequence_number, filter_operation->is_forward_pass);
		}
	}

	string table_name = filter_operation ? "table_" + std::to_string(filter_operation->probe_table_idx) : "unknown";

	auto &state = state_p.Cast<PhysicalProbeFilterState>();

	// lazy initialization of bloom filters on first call
	if (!state.bloom_filters_initialized) {
		D_PRINTF("[EXEC_INTERNAL] PROBE_FILTER (probe=%s) Initializing bloom filters, bound_column_indices.size()=%zu",
		         table_name.c_str(), bound_column_indices.size());
		for (size_t i = 0; i < bound_column_indices.size(); i++) {
			D_PRINTF("  bound_column_indices[%zu] = %llu", i, (unsigned long long)bound_column_indices[i]);
		}

		if (!related_create_filter_vec.empty() && filter_operation) {
			// lookup bloom filters by build column binding
			for (const auto &build_col : filter_operation->build_columns) {
				for (auto *create_filter : related_create_filter_vec) {
					auto bf = create_filter->GetBloomFilter(build_col);
					if (bf) {
						string build_table =
						    create_filter->filter_operation
						        ? "table_" + std::to_string(create_filter->filter_operation->build_table_idx)
						        : "unknown";
						D_PRINTF("[EXEC_INTERNAL] PROBE_FILTER found bloom filter for col(%llu,%llu) from "
						         "CREATE_FILTER (build=%s)",
						         (unsigned long long)build_col.table_index, (unsigned long long)build_col.column_index,
						         build_table.c_str());
						state.bloom_filters.push_back(bf);
						state.sel_vector.emplace_back(STANDARD_VECTOR_SIZE);
						break; // found the filter for this column
					}
				}
			}
		}
		D_PRINTF("[EXEC_INTERNAL] PROBE_FILTER total bloom_filters.size() = %zu", state.bloom_filters.size());
		state.bloom_filters_initialized = true;
	}

	idx_t row_num = input.size();

	// if no bloom filters or no input, just pass through
	if (state.bloom_filters.empty() || row_num == 0) {
		D_PRINTF("[EXEC_INTERNAL] PROBE_FILTER (probe=%s) No bloom filter input/empty, row_num = %llu",
		         table_name.c_str(), (unsigned long long)row_num);
		if (profiling_stats) {
			profiling_stats->rows_in.fetch_add(row_num, std::memory_order_relaxed);
			profiling_stats->rows_out.fetch_add(row_num, std::memory_order_relaxed);
		}
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	idx_t original_row_num = row_num;

	// apply bloom filters
	idx_t result_count = row_num;

	unique_ptr<ScopedTimer> probe_timer;
	if (profiling_stats) {
		probe_timer = make_uniq<ScopedTimer>(profiling_stats->probe_time_us);
	}

	for (int i = 0; i < state.bloom_filters.size(); i++) {
		auto bf = state.bloom_filters[i];
		if (!bf || !bf->finalized_) {
			D_PRINT("skipped - bloom filter not ready");
			continue;
		}

		// check if bloom filter is empty (no data inserted)
		if (bf->IsEmpty()) {
			string build_table =
			    filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
			D_PRINTF("Bloom filter empty for %s", build_table.c_str());
			// signal any CREATE_FILTER siblings targeting this probe that it will be empty
			auto reg = GetProbeEmptyRegistry(context.client);
			if (reg) {
				auto flag = reg->GetOrCreate(filter_operation->probe_table_idx);
				flag->store(true, std::memory_order_relaxed);
			}
			// empty filter means no matches possible
			probe_timer.reset();
			if (profiling_stats) {
				profiling_stats->rows_in.fetch_add(original_row_num, std::memory_order_relaxed);
			}
			chunk.SetCardinality(0);
			return OperatorResultType::NEED_MORE_INPUT;
		}

		// string probe_table = filter_operation ? "table_" + std::to_string(filter_operation->probe_table_idx) :
		// "unknown"; for (int i = 0; i < bound_column_indices.size(); i++) { 	printf("bound columns for %s - %llu\n",
		// probe_table.c_str(), bound_column_indices[i]);
		// }

		// lookup directly into selection vector

		result_count = bf->LookupSel(input, state.sel_vector[i], {bound_column_indices[i]}, state.bit_vector.data());

		// early exit if no rows passed
		if (result_count == 0) {
			probe_timer.reset();
			if (profiling_stats) {
				profiling_stats->rows_in.fetch_add(original_row_num, std::memory_order_relaxed);
			}
			chunk.SetCardinality(0);
			return OperatorResultType::NEED_MORE_INPUT;
		}

		// apply filter if we filtered rows
		if (result_count < row_num) {
			input.Slice(state.sel_vector[i], result_count);
			row_num = result_count;
		}
	}

	// stop probe timer before output work
	probe_timer.reset();

	chunk.Reference(input);

	if (profiling_stats) {
		profiling_stats->rows_in.fetch_add(original_row_num, std::memory_order_relaxed);
		profiling_stats->rows_out.fetch_add(result_count, std::memory_order_relaxed);
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

void PhysicalProbeFilter::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

#ifdef DEBUG
	char ptr_str[32];
	snprintf(ptr_str, sizeof(ptr_str), "%p", (void *)this);
	string probe_table = filter_operation ? "table_" + std::to_string(filter_operation->probe_table_idx) : "unknown";
	Printer::Print(StringUtil::Format("[PIPELINE] PROBE_FILTER (probe=%s, this=%s) BuildPipelines called",
	                                  probe_table.c_str(), ptr_str));
#endif

	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);

#ifdef DEBUG
	Printer::Print(
	    StringUtil::Format("[PIPELINE] PROBE_FILTER (probe=%s, this=%s) added to current pipeline as operator",
	                       probe_table.c_str(), ptr_str));
	Printer::Print(StringUtil::Format("[PIPELINE] PROBE_FILTER (probe=%s) has %zu related CREATE_FILTER operators",
	                                  probe_table.c_str(), related_create_filter_vec.size()));
#endif

	// add dependencies on all related CREATE_FILTER operators
	for (size_t i = 0; i < related_create_filter_vec.size(); i++) {
		auto *create_filter = related_create_filter_vec[i];
		create_filter->BuildPipelinesFromRelated(current, meta_pipeline);
	}

	// continue building child pipelines
	children[0].get().BuildPipelines(current, meta_pipeline);
}

} // namespace duckdb
