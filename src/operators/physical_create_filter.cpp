#include "physical_create_filter.hpp"
#include "bloom_filter.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "utils/debug_utils.hpp"
#include "robust_profiling.hpp"
#include "probe_empty_registry.hpp"
#include <duckdb/parallel/meta_pipeline.hpp>
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

PhysicalCreateFilter::PhysicalCreateFilter(PhysicalPlan &physical_plan,
                                           const shared_ptr<FilterOperation> &filter_operation,
                                           vector<LogicalType> types, idx_t estimated_cardinality,
                                           vector<idx_t> bound_column_indices)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      filter_operation(filter_operation), is_probing_side(false),
      bound_column_indices(std::move(bound_column_indices)) {
	// create bloom filter for each build column, keyed by ColumnBinding
	for (size_t i = 0; i < filter_operation->build_columns.size(); i++) {
		const auto &col = filter_operation->build_columns[i];
		bloom_filter_map[col] = make_shared_ptr<PTBloomFilter>();
	}
}

string PhysicalCreateFilter::GetName() const {
	return "CREATE_FILTER";
}

string PhysicalCreateFilter::ToString(optional_ptr<ClientContext> context, const ProfilerPrintFormat &format) const {
	string result = "CREATE_FILTER";
	result += " [" + std::to_string(filter_operation->build_columns.size()) + " filters]";
	return result;
}

InsertionOrderPreservingMap<string> PhysicalCreateFilter::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Operator"] = "PhysicalCreateFilter";
	result["Build Table"] = to_string(filter_operation->build_table_idx);
	// there can be multiple probe tables for a single create
	string probe_tables;
	vector<idx_t> seen_probe;
	for (const auto &col : filter_operation->probe_columns) {
		bool found = false;
		for (auto idx : seen_probe) {
			if (idx == col.table_index.index) {
				found = true;
				break;
			}
		}
		if (!found) {
			if (!probe_tables.empty()) {
				probe_tables += ", ";
			}
			probe_tables += to_string(col.table_index.index);
			seen_probe.push_back(col.table_index.index);
		}
	}
	result["Probe Tables"] = probe_tables;

	string build_cols = "";
	for (size_t i = 0; i < filter_operation->build_columns.size(); i++) {
		if (i > 0) {
			build_cols += ", ";
		}
		build_cols += "(" + to_string(filter_operation->build_columns[i].table_index.index) + "." +
		              to_string(filter_operation->build_columns[i].column_index) + ")";
	}
	result["Build Columns"] = build_cols;

	if (estimated_cardinality != DConstants::INVALID_INDEX) {
		result["Estimated Cardinality"] = std::to_string(estimated_cardinality);
	}

	return result;
}

//===--------------------------------------------------------------------===//
// Min-Max helpers
//===--------------------------------------------------------------------===//

template <typename T>
static void TypedUpdateMinMax(Vector &vec, ColumnMinMax &mm) {
	T local_min {}, local_max {};
	bool has_val = false;

	for (auto entry : vec.Values<T>()) {
		if (!entry.IsValid()) {
			continue;
		}
		const auto &val = entry.GetValue();
		if (!has_val) {
			local_min = val;
			local_max = val;
			has_val = true;
		} else {
			if (val < local_min) {
				local_min = val;
			}
			if (val > local_max) {
				local_max = val;
			}
		}
	}

	if (!has_val) {
		return;
	}

	Value vmin = Value::CreateValue(local_min);
	Value vmax = Value::CreateValue(local_max);
	if (!mm.has_value) {
		mm.min_val = vmin;
		mm.max_val = vmax;
		mm.has_value = true;
	} else {
		if (vmin < mm.min_val) {
			mm.min_val = vmin;
		}
		if (vmax > mm.max_val) {
			mm.max_val = vmax;
		}
	}
}

static void UpdateMinMax(Vector &vec, ColumnMinMax &mm) {
	auto &type = vec.GetType();
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		TypedUpdateMinMax<int8_t>(vec, mm);
		break;
	case LogicalTypeId::SMALLINT:
		TypedUpdateMinMax<int16_t>(vec, mm);
		break;
	case LogicalTypeId::INTEGER:
		TypedUpdateMinMax<int32_t>(vec, mm);
		break;
	case LogicalTypeId::BIGINT:
		TypedUpdateMinMax<int64_t>(vec, mm);
		break;
	case LogicalTypeId::UTINYINT:
		TypedUpdateMinMax<uint8_t>(vec, mm);
		break;
	case LogicalTypeId::USMALLINT:
		TypedUpdateMinMax<uint16_t>(vec, mm);
		break;
	case LogicalTypeId::UINTEGER:
		TypedUpdateMinMax<uint32_t>(vec, mm);
		break;
	case LogicalTypeId::UBIGINT:
		TypedUpdateMinMax<uint64_t>(vec, mm);
		break;
	case LogicalTypeId::FLOAT:
		TypedUpdateMinMax<float>(vec, mm);
		break;
	case LogicalTypeId::DOUBLE:
		TypedUpdateMinMax<double>(vec, mm);
		break;
	case LogicalTypeId::DATE:
		TypedUpdateMinMax<date_t>(vec, mm);
		break;
	case LogicalTypeId::TIMESTAMP:
		TypedUpdateMinMax<timestamp_t>(vec, mm);
		break;
	case LogicalTypeId::VARCHAR:
		TypedUpdateMinMax<string_t>(vec, mm);
		break;
	default:
		break;
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//

CreateFilterGlobalSinkState::CreateFilterGlobalSinkState(ClientContext &context, const PhysicalCreateFilter &op)
    : op(op) {
	total_data = make_uniq<ColumnDataCollection>(context, op.types);
	// resolve shared probe-empty flag once (single-threaded); forward pass only
	if (op.is_forward_pass) {
		auto reg = GetProbeEmptyRegistry(context);
		if (reg) {
			probe_empty_flag = reg->GetOrCreate(op.filter_operation->probe_table_idx);
		}
		Value v;
		if (context.TryGetCurrentSetting("robust_dynamic_or_filter_threshold", v)) {
			distinct_threshold = v.GetValue<uint64_t>();
		}
		column_distinct.resize(op.bound_column_indices.size());
	}
}

CreateFilterLocalSinkState::CreateFilterLocalSinkState(ClientContext &context, const PhysicalCreateFilter &op)
    : client_context(context) {
	local_data = make_uniq<ColumnDataCollection>(client_context, op.types);
	// initialize min-max and distinct tracking for each build column
	if (op.is_forward_pass) {
		local_min_max.resize(op.bound_column_indices.size());
		local_distinct.resize(op.bound_column_indices.size());
	}
}

SinkResultType PhysicalCreateFilter::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	if (!profiling_checked) {
		profiling_checked = true;
		auto prof = GetRobustProfilingState(context.client);
		if (prof) {
			profiling_stats =
			    prof->RegisterCreateFilter(filter_operation->build_table_idx, filter_operation->probe_columns,
			                               filter_operation->sequence_number, is_forward_pass);
		}
	}

	// short-circuit: if probe side is already known empty, don't ingest into BF.
	// single relaxed load, lock-free.
	auto &gstate = input.global_state.Cast<CreateFilterGlobalSinkState>();
	if (gstate.probe_empty_flag && gstate.probe_empty_flag->load(std::memory_order_relaxed)) {
		return SinkResultType::FINISHED;
	}

	CreateFilterLocalSinkState &local_state = input.local_state.Cast<CreateFilterLocalSinkState>();
	if (profiling_stats) {
		ScopedCpuTimer timer(profiling_stats->sink_time_us);
		profiling_stats->rows_materialized.fetch_add(chunk.size(), std::memory_order_relaxed);
		local_state.local_data->Append(chunk);
	} else {
		local_state.local_data->Append(chunk);
	}

	// compute min-max using typed pointer access
	if (is_forward_pass && !local_state.local_min_max.empty() && chunk.size() > 0) {
		for (idx_t i = 0; i < bound_column_indices.size() && i < local_state.local_min_max.size(); i++) {
			idx_t col_idx = bound_column_indices[i];
			if (col_idx >= chunk.ColumnCount()) {
				continue;
			}
			auto &vec = chunk.data[col_idx];
			UpdateMinMax(vec, local_state.local_min_max[i]);
		}
	}

	// track distinct values up to threshold+1 (overflow stops further insertion)
	if (is_forward_pass && !local_state.local_distinct.empty() && chunk.size() > 0) {
		const idx_t threshold = gstate.distinct_threshold;
		for (idx_t i = 0; i < bound_column_indices.size() && i < local_state.local_distinct.size(); i++) {
			auto &cd = local_state.local_distinct[i];
			if (cd.over_threshold) {
				continue;
			}
			idx_t col_idx = bound_column_indices[i];
			if (col_idx >= chunk.ColumnCount()) {
				continue;
			}
			auto &vec = chunk.data[col_idx];
			for (idx_t row = 0; row < chunk.size(); row++) {
				Value val = vec.GetValue(row);
				if (val.IsNull()) {
					continue;
				}
				cd.values.insert(std::move(val));
				if (cd.values.size() > threshold) {
					cd.over_threshold = true;
					cd.values.clear();
					break;
				}
			}
		}
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateFilter::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	CreateFilterGlobalSinkState &gstate = input.global_state.Cast<CreateFilterGlobalSinkState>();
	CreateFilterLocalSinkState &local_state = input.local_state.Cast<CreateFilterLocalSinkState>();
	lock_guard<mutex> lock(gstate.glock);
	gstate.local_data_collections.emplace_back(std::move(local_state.local_data));

	// merge local min-max into global
	if (!local_state.local_min_max.empty()) {
		if (gstate.column_min_max.empty()) {
			gstate.column_min_max.resize(local_state.local_min_max.size());
		}
		for (idx_t i = 0; i < local_state.local_min_max.size(); i++) {
			auto &local_mm = local_state.local_min_max[i];
			if (!local_mm.has_value) {
				continue;
			}
			auto &global_mm = gstate.column_min_max[i];
			if (!global_mm.has_value) {
				global_mm = local_mm;
			} else {
				if (local_mm.min_val < global_mm.min_val) {
					global_mm.min_val = local_mm.min_val;
				}
				if (local_mm.max_val > global_mm.max_val) {
					global_mm.max_val = local_mm.max_val;
				}
			}
		}
	}

	// merge local distinct values into global, propagating over_threshold
	if (!local_state.local_distinct.empty()) {
		const idx_t threshold = gstate.distinct_threshold;
		for (idx_t i = 0; i < local_state.local_distinct.size() && i < gstate.column_distinct.size(); i++) {
			auto &local_cd = local_state.local_distinct[i];
			auto &global_cd = gstate.column_distinct[i];
			if (global_cd.over_threshold) {
				continue;
			}
			if (local_cd.over_threshold) {
				global_cd.over_threshold = true;
				global_cd.values.clear();
				continue;
			}
			for (auto &val : local_cd.values) {
				global_cd.values.insert(val);
				if (global_cd.values.size() > threshold) {
					global_cd.over_threshold = true;
					global_cd.values.clear();
					break;
				}
			}
		}
	}

	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

// pushes dynamic filters (BF + min-max) to table scans after BF is fully built
static void PushDynamicFilters(const PhysicalCreateFilter &op, const CreateFilterGlobalSinkState &gsink,
                               ClientContext &context) {
	if (!op.is_forward_pass || op.pushdown_targets.empty()) {
		return;
	}

	// if build side produced 0 rows, no probe-side rows can match — push always-false filter
	if (gsink.total_data->Count() == 0) {
		for (auto &target : op.pushdown_targets) {
			// Keep a comparison expression so zonemap pruning can skip every row group.
			auto always_false = make_uniq<ExpressionFilter>(BoundComparisonExpression::Create(
			    ExpressionType::COMPARE_EQUAL, make_uniq<BoundConstantExpression>(Value::INTEGER(0)),
			    make_uniq<BoundConstantExpression>(Value::INTEGER(1))));
			target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(always_false));
			D_PRINTF("[PUSHDOWN] pushed always-false for col %s (empty build side)", target.column_name.c_str());
		}
		return;
	}

	string filter_type = "all";
	Value filter_type_val;
	if (context.TryGetCurrentSetting("robust_filter_type", filter_type_val)) {
		filter_type = filter_type_val.GetValue<string>();
	}

	bool push_bf = (filter_type == "all" || filter_type == "bf_only");
	bool push_minmax = (filter_type == "all" || filter_type == "minmax_only");
	bool consider_in = (filter_type == "all");

	for (auto &target : op.pushdown_targets) {
		for (size_t i = 0; i < op.filter_operation->build_columns.size(); i++) {
			if (i >= op.filter_operation->probe_columns.size()) {
				break;
			}
			const auto &probe_col = op.filter_operation->probe_columns[i];
			if (probe_col.table_index.index != target.probe_column.table_index.index ||
			    probe_col.column_index != target.probe_column.column_index) {
				continue;
			}

			const auto &build_col = op.filter_operation->build_columns[i];

			// push IN-filter (zonemap-only) or equality constant alongside BF; equality
			// is per-row and supersedes BF/min-max
			bool pushed_equal = false;
			if (consider_in && i < gsink.column_distinct.size()) {
				auto &cd = gsink.column_distinct[i];
				if (!cd.over_threshold && cd.values.size() == 1) {
					auto eq =
					    ExpressionFilter::CreateComparisonFilter(ExpressionType::COMPARE_EQUAL, *cd.values.begin());
					target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(eq));
					pushed_equal = true;
					D_PRINTF("[PUSHDOWN] pushed equality constant for col %s", target.column_name.c_str());
				} else if (!cd.over_threshold && cd.values.size() > 1) {
					vector<Value> in_list(cd.values.begin(), cd.values.end());
					if (!FilterCombiner::ContainsNull(in_list) && !FilterCombiner::IsDenseRange(in_list)) {
						auto in_expr = ExpressionFilter::CreateInExpression(
						    make_uniq<BoundReferenceExpression>(target.column_type, 0ULL), std::move(in_list));
						auto opt = make_uniq<ExpressionFilter>(
						    CreateOptionalFilterExpression(std::move(in_expr), target.column_type));
						target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(opt));
						D_PRINTF("[PUSHDOWN] pushed IN-filter (%llu values) for col %s",
						         (unsigned long long)cd.values.size(), target.column_name.c_str());
					}
				}
			}

			// keep BF alongside IN-filter for per-row pruning (IN is zonemap-only);
			// equality filter already does per-row, skip BF there
			if (push_bf && !pushed_equal) {
				auto bf_it = op.bloom_filter_map.find(build_col);
				if (bf_it != op.bloom_filter_map.end() && bf_it->second && !bf_it->second->IsEmpty()) {
					vector<unique_ptr<Expression>> arguments;
					arguments.push_back(make_uniq<BoundReferenceExpression>(target.column_type, 0ULL));
					// Keep scan-side Bloom checks active by disabling selectivity tracking.
					auto bf_expr = make_uniq<BoundFunctionExpression>(
					    BoundScalarFunction(BloomFilterScalarFun::GetFunction(target.column_type)),
					    std::move(arguments),
					    make_uniq<BloomFilterFunctionData>(bf_it->second->GetNativeFilter(), false, target.column_name,
					                                       target.column_type, 0.0f, idx_t(0)));
					auto bf_filter = make_uniq<ExpressionFilter>(std::move(bf_expr));
					target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(bf_filter));
					D_PRINTF("[PUSHDOWN] pushed BF for col %s to scan col %llu", target.column_name.c_str(),
					         (unsigned long long)target.scan_column_index);
				}
			}

			// equality filter already expresses min/max; skip min/max push in that case
			if (push_minmax && !pushed_equal && i < gsink.column_min_max.size() && gsink.column_min_max[i].has_value) {
				auto &mm = gsink.column_min_max[i];
				auto min_filter =
				    ExpressionFilter::CreateComparisonFilter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, mm.min_val);
				target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(min_filter));

				auto max_filter =
				    ExpressionFilter::CreateComparisonFilter(ExpressionType::COMPARE_LESSTHANOREQUALTO, mm.max_val);
				target.dynamic_filters->PushFilter(op, target.scan_column_index, std::move(max_filter));

				D_PRINTF("[PUSHDOWN] pushed min-max for col %s [%s, %s]", target.column_name.c_str(),
				         mm.min_val.ToString().c_str(), mm.max_val.ToString().c_str());
			}

			break;
		}
	}
}

// parallelize the hashing/insert pass only when the build side is large enough to amortize
// task-scheduling overhead; below this an inline single-threaded build is faster.
static constexpr idx_t PARALLEL_BUILD_THRESHOLD = 102400; // ~100k rows (50 vectors)

void PhysicalCreateFilter::PostBuildFinalize(const CreateFilterGlobalSinkState &gsink, ClientContext &context) const {
	const idx_t actual_rows = gsink.total_data->Count();

	for (auto &entry : bloom_filter_map) {
		if (entry.second) {
			entry.second->Seal(context);
		}
	}

	// if this forward CREATE_FILTER produced an empty BF, signal sibling CREATE_FILTERs targeting
	// the same probe that the probe side will be empty (relaxed store, lock-free).
	if (gsink.probe_empty_flag && actual_rows == 0) {
		gsink.probe_empty_flag->store(true, std::memory_order_relaxed);
	}

	PushDynamicFilters(*this, gsink, context);
}

namespace {

// hashes + inserts a disjoint range of total_data chunks into the (already-allocated) BFs.
// native BF inserts use atomic fetch_or, so concurrent inserts into the same filter are safe.
class CreateFilterBuildTask : public ExecutorTask {
public:
	CreateFilterBuildTask(Pipeline &pipeline, shared_ptr<Event> event, const PhysicalCreateFilter &op,
	                      CreateFilterGlobalSinkState &gsink, idx_t chunk_from, idx_t chunk_to)
	    : ExecutorTask(pipeline.GetClientContext(), std::move(event), op), op(op), gsink(gsink), chunk_from(chunk_from),
	      chunk_to(chunk_to) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		DataChunk chunk;
		gsink.total_data->InitializeScanChunk(chunk);
		for (idx_t cid = chunk_from; cid < chunk_to; cid++) {
			gsink.total_data->FetchChunk(cid, chunk);
			for (size_t i = 0; i < op.filter_operation->build_columns.size(); i++) {
				auto it = op.bloom_filter_map.find(op.filter_operation->build_columns[i]);
				if (it != op.bloom_filter_map.end() && it->second) {
					it->second->Insert(chunk, {op.bound_column_indices[i]});
				}
			}
		}
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

	string TaskType() const override {
		return "CreateFilterBuildTask";
	}

private:
	const PhysicalCreateFilter &op;
	CreateFilterGlobalSinkState &gsink;
	idx_t chunk_from;
	idx_t chunk_to;
};

class CreateFilterBuildEvent : public BasePipelineEvent {
public:
	CreateFilterBuildEvent(Pipeline &pipeline, const PhysicalCreateFilter &op, CreateFilterGlobalSinkState &gsink)
	    : BasePipelineEvent(pipeline), op(op), gsink(gsink) {
	}

	void Schedule() override {
		auto &client = pipeline->GetClientContext();
		const idx_t num_threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(client).NumberOfThreads());
		const idx_t chunk_count = gsink.total_data->ChunkCount();
		const idx_t num_tasks = MaxValue<idx_t>(MinValue<idx_t>(num_threads, chunk_count), 1);
		const idx_t chunks_per_task = (chunk_count + num_tasks - 1) / num_tasks;

		vector<shared_ptr<Task>> tasks;
		for (idx_t cid = 0; cid < chunk_count;) {
			const idx_t from = cid;
			const idx_t to = MinValue<idx_t>(from + chunks_per_task, chunk_count);
			tasks.push_back(make_uniq<CreateFilterBuildTask>(*pipeline, shared_from_this(), op, gsink, from, to));
			cid = to;
		}
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		// runs after all build tasks complete, before dependent (probe) pipelines run
		op.PostBuildFinalize(gsink, pipeline->GetClientContext());
	}

private:
	const PhysicalCreateFilter &op;
	CreateFilterGlobalSinkState &gsink;
};

} // namespace

SinkFinalizeType PhysicalCreateFilter::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                OperatorSinkFinalizeInput &input) const {
	// lazy init profiling if Sink was never called (e.g., empty input)
	if (!profiling_checked) {
		profiling_checked = true;
		auto prof = GetRobustProfilingState(context);
		if (prof) {
			profiling_stats =
			    prof->RegisterCreateFilter(filter_operation->build_table_idx, filter_operation->probe_columns,
			                               filter_operation->sequence_number, is_forward_pass);
		}
	}

	auto &gsink = input.global_state.Cast<CreateFilterGlobalSinkState>();

	// time the finalize phase
	unique_ptr<ScopedTimer> fin_timer;
	if (profiling_stats) {
		fin_timer = make_uniq<ScopedTimer>(profiling_stats->finalize_time_us);
	}

	// 1. merge local data collections - needed for downstream Source
	for (auto &local_data : gsink.local_data_collections) {
		gsink.total_data->Combine(*local_data);
	}
	gsink.local_data_collections.clear();

	string build_table = filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
	D_PRINTF("[FINALIZE] CREATE_FILTER (build=%s): total_data contains %llu rows, %zu bloom filters",
	         build_table.c_str(), (unsigned long long)gsink.total_data->Count(), bloom_filter_map.size());

	// 2. build bloom filters from the materialized data, sized to the actual row count
	const idx_t actual_rows = gsink.total_data->Count();

	// large build side: allocate up front (sized to actual rows), then hash/insert in parallel via
	// scheduled tasks; post-build steps run from the event's completion.
	if (actual_rows >= PARALLEL_BUILD_THRESHOLD && gsink.total_data->ChunkCount() > 1) {
		for (size_t i = 0; i < filter_operation->build_columns.size(); i++) {
			auto it = bloom_filter_map.find(filter_operation->build_columns[i]);
			if (it != bloom_filter_map.end() && it->second) {
				it->second->Initialize(context, actual_rows);
			}
		}
		fin_timer.reset(); // insert time is spent on the scheduled tasks, not this thread
		event.InsertEvent(make_shared_ptr<CreateFilterBuildEvent>(pipeline, *this, gsink));
		return SinkFinalizeType::READY;
	}

	// small (or empty) build side: build inline on this thread
	if (actual_rows > 0) {
		for (size_t i = 0; i < filter_operation->build_columns.size(); i++) {
			const auto &col = filter_operation->build_columns[i];
			auto it = bloom_filter_map.find(col);
			if (it == bloom_filter_map.end() || !it->second) {
				continue;
			}
			it->second->ReinitializeAndRehash(context, actual_rows, *gsink.total_data, {bound_column_indices[i]});
		}
	}

	PostBuildFinalize(gsink, context);
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSinkState> PhysicalCreateFilter::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CreateFilterGlobalSinkState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalCreateFilter::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreateFilterLocalSinkState>(context.client, *this);
}

shared_ptr<PTBloomFilter> PhysicalCreateFilter::GetBloomFilter(const ColumnBinding &col) const {
	auto it = bloom_filter_map.find(col);
	if (it != bloom_filter_map.end()) {
		return it->second;
	}
	return nullptr;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

CreateFilterGlobalSourceState::CreateFilterGlobalSourceState(ClientContext &context, const PhysicalCreateFilter &op)
    : context(context) {
	D_ASSERT(op.sink_state);
	auto &gstate = op.sink_state->Cast<CreateFilterGlobalSinkState>();
	gstate.total_data->InitializeScan(scan_state);
	partition_id = 0;
}

idx_t CreateFilterGlobalSourceState::MaxThreads() {
	return TaskScheduler::GetScheduler(context).NumberOfThreads();
}

unique_ptr<GlobalSourceState> PhysicalCreateFilter::GetGlobalSourceState(ClientContext &context) const {
	auto state = make_uniq<CreateFilterGlobalSourceState>(context, *this);

	D_ASSERT(sink_state);
	auto &gsink = sink_state->Cast<CreateFilterGlobalSinkState>();

	auto chunk_count = gsink.total_data->ChunkCount();
	auto row_count = gsink.total_data->Count();

#ifdef DEBUG
	string build_table = filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
	Printer::Print(
	    StringUtil::Format("[SOURCE] CREATE_FILTER (build=%s) GetGlobalSourceState: chunk_count=%llu, row_count=%llu",
	                       build_table.c_str(), (unsigned long long)chunk_count, (unsigned long long)row_count));
#endif

	const idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	auto chunks_per_thread = MaxValue<idx_t>((chunk_count + num_threads - 1) / num_threads, 1);
	idx_t chunk_idx = 0;
	for (idx_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
		if (chunk_idx == chunk_count) {
			break;
		}
		auto chunk_idx_from = chunk_idx;
		auto chunk_idx_to = MinValue<idx_t>(chunk_idx_from + chunks_per_thread, chunk_count);
		state->chunks_todo.emplace_back(chunk_idx_from, chunk_idx_to);
#ifdef DEBUG
		Printer::Print(StringUtil::Format("[SOURCE] CREATE_FILTER (build=%s) Partition %llu: chunks [%llu, %llu)",
		                                  build_table.c_str(), (unsigned long long)thread_idx,
		                                  (unsigned long long)chunk_idx_from, (unsigned long long)chunk_idx_to));
#endif
		chunk_idx = chunk_idx_to;
	}
	return unique_ptr_cast<CreateFilterGlobalSourceState, GlobalSourceState>(std::move(state));
}

unique_ptr<LocalSourceState> PhysicalCreateFilter::GetLocalSourceState(ExecutionContext &context,
                                                                       GlobalSourceState &gstate) const {
	return make_uniq<CreateFilterLocalSourceState>();
}

SourceResultType PhysicalCreateFilter::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                       OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<CreateFilterGlobalSinkState>();
	auto &lstate = input.local_state.Cast<CreateFilterLocalSourceState>();
	auto &state = input.global_state.Cast<CreateFilterGlobalSourceState>();

#ifdef DEBUG
	string build_table = filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
#endif

	if (lstate.initial) {
		lstate.local_partition_id = state.partition_id.fetch_add(1);
		lstate.initial = false;

#ifdef DEBUG
		Printer::Print(StringUtil::Format(
		    "[SOURCE] CREATE_FILTER (build=%s) GetData initial: partition_id=%llu, chunks_todo.size()=%zu",
		    build_table.c_str(), (unsigned long long)lstate.local_partition_id, state.chunks_todo.size()));
#endif

		if (lstate.local_partition_id >= state.chunks_todo.size()) {
			D_PRINTF("[SOURCE] CREATE_FILTER No more partitions, returning FINISHED");
			return SourceResultType::FINISHED;
		}
		lstate.chunk_from = state.chunks_todo[lstate.local_partition_id].first;
		lstate.chunk_to = state.chunks_todo[lstate.local_partition_id].second;

		// parallel source
		lstate.local_current_chunk_id = lstate.chunk_from;

#ifdef DEBUG
		Printer::Print(StringUtil::Format("[SOURCE] CREATE_FILTER (build=%s) Assigned range: [%llu, %llu)",
		                                  build_table.c_str(), (unsigned long long)lstate.chunk_from,
		                                  (unsigned long long)lstate.chunk_to));
#endif
	}

	// sequential source
	// auto chunk_count = gstate.total_data->ChunkCount();
	//
	// if (lstate.local_current_chunk_id >= chunk_count) {
	// 	return SourceResultType::FINISHED;
	// }
	//
	// if (lstate.local_current_chunk_id == 0) {
	// 	lstate.local_current_chunk_id = lstate.chunk_from;
	// }

	// parallel source
	{
		// auto chunk_count = gstate.total_data->ChunkCount();

		if (lstate.local_current_chunk_id >= lstate.chunk_to) {
			return SourceResultType::FINISHED;
		}
	}
	if (profiling_stats) {
		ScopedTimer timer(profiling_stats->source_time_us);
		gstate.total_data->FetchChunk(lstate.local_current_chunk_id++, chunk);
	} else {
		gstate.total_data->FetchChunk(lstate.local_current_chunk_id++, chunk);
	}
	return SourceResultType::HAVE_MORE_OUTPUT;
}

void PhysicalCreateFilter::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

#ifdef DEBUG
	string build_table = filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
#endif

	auto &state = meta_pipeline.GetState();

	// make this operator source of the pipeline
	state.SetPipelineSource(current, *this);

	if (this_pipeline == nullptr) {
		// D_PRINTF("[PIPELINE] CREATE_FILTER (build=%s) creating NEW child pipeline for build-side",
		// build_table.c_str());
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		this_pipeline = child_meta_pipeline.GetBasePipeline();
		// CreateChildMetaPipeline() automatically registers the child pipeline as a dependency
		child_meta_pipeline.Build(children[0].get());
		// D_PRINTF("[PIPELINE] CREATE_FILTER (build=%s) child pipeline created", build_table.c_str());
	} else {
		// D_PRINTF("[PIPELINE] CREATE_FILTER (build=%s) adding existing child pipeline as dependency",
		// build_table.c_str());
		current.AddDependency(this_pipeline);
	}
}

void PhysicalCreateFilter::BuildPipelinesFromRelated(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	D_ASSERT(children.size() == 1);

#ifdef DEBUG
	string build_table = filter_operation ? "table_" + std::to_string(filter_operation->build_table_idx) : "unknown";
	char ptr_str[32];
	snprintf(ptr_str, sizeof(ptr_str), "%p", (void *)this);
	Printer::Print(StringUtil::Format(
	    "[PIPELINE] CREATE_FILTER (build=%s, this=%s) BuildPipelinesFromRelated - PROBE_FILTER needs this filter",
	    build_table.c_str(), ptr_str));
#endif

	if (this_pipeline == nullptr) {
		D_PRINTF("[PIPELINE] CREATE_FILTER creating NEW child pipeline from BuildPipelinesFromRelated");
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		this_pipeline = child_meta_pipeline.GetBasePipeline();
		child_meta_pipeline.Build(children[0].get());
		D_PRINT("[PIPELINE] CREATE_FILTER child pipeline created and dependency added automatically");
	} else {
		D_PRINT("[PIPELINE] CREATE_FILTER adding existing pipeline as dependency");
		current.AddDependency(this_pipeline);
	}

#ifdef DEBUG
	this_pipeline->Print();
#endif
}

} // namespace duckdb
