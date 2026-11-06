#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "bloom_filter.hpp"
#include "../optimizer/graph_manager.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include <duckdb/common/types/column/column_data_scan_states.hpp>
#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/common/types/value_map.hpp"

namespace duckdb {

struct CreateFilterStats;
class CreateFilterGlobalSinkState;

class PhysicalCreateFilter : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalCreateFilter(PhysicalPlan &physical_plan, const shared_ptr<FilterOperation> &filter_operation,
	                     vector<LogicalType> types, idx_t estimated_cardinality, vector<idx_t> bound_column_indices);

	// Required virtual methods
	~PhysicalCreateFilter() override = default;

	string GetName() const override;
	string ToString(optional_ptr<ClientContext> context = nullptr,
	                const ProfilerPrintFormat &format = ProfilerPrintFormat::Default()) const override;

	// populate info in query plan
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// sink interface - PhysicalOperator can act as sink
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	// mark filters finalized, signal empty probe, push dynamic filters; runs inline for small
	// builds or from the parallel build event's completion for large builds.
	void PostBuildFinalize(const CreateFilterGlobalSinkState &gsink, ClientContext &context) const;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}
	// source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	void BuildPipelinesFromRelated(Pipeline &current, MetaPipeline &meta_pipeline);

public:
	// vector<shared_ptr<FilterPlan>> filter_plans;
	shared_ptr<FilterOperation> filter_operation;
	bool is_probing_side;

	// maps the column indices to resolved chunk column positions
	vector<idx_t> bound_column_indices;

	// column-keyed bloom filter map: ColumnBinding -> PTBloomFilter
	unordered_map<ColumnBinding, shared_ptr<PTBloomFilter>, ColumnBindingHash, ColumnBindingEqual> bloom_filter_map;

	// pipeline reference
	shared_ptr<Pipeline> this_pipeline;

	// lookup bloom filter by the column it was built on
	shared_ptr<PTBloomFilter> GetBloomFilter(const ColumnBinding &col) const;

	// dynamic filter pushdown to table scans (forward pass only)
	struct DynamicFilterTarget {
		shared_ptr<DynamicTableFilterSet> dynamic_filters;
		ProjectionIndex scan_column_index;
		ColumnBinding probe_column;
		LogicalType column_type;
		string column_name;
	};
	vector<DynamicFilterTarget> pushdown_targets;
	bool is_forward_pass = false;

	// profiling
	mutable shared_ptr<CreateFilterStats> profiling_stats;
	mutable bool profiling_checked = false;
};

struct ColumnMinMax {
	Value min_val, max_val;
	bool has_value = false;
};

struct ColumnDistinct {
	value_set_t values;
	bool over_threshold = false;
};

class CreateFilterLocalSinkState : public LocalSinkState {
public:
	CreateFilterLocalSinkState(ClientContext &context, const PhysicalCreateFilter &op);

	ClientContext &client_context;
	unique_ptr<ColumnDataCollection> local_data;
	vector<ColumnMinMax> local_min_max;
	vector<ColumnDistinct> local_distinct;
};

class CreateFilterGlobalSinkState : public GlobalSinkState {
public:
	CreateFilterGlobalSinkState(ClientContext &context, const PhysicalCreateFilter &op);

	const PhysicalCreateFilter &op;
	mutex glock;

	// store data for sink phase
	unique_ptr<ColumnDataCollection> total_data;
	vector<unique_ptr<ColumnDataCollection>> local_data_collections;

	// min-max tracking for dynamic filter pushdown
	vector<ColumnMinMax> column_min_max;

	// distinct-value tracking for IN-filter pushdown
	vector<ColumnDistinct> column_distinct;
	idx_t distinct_threshold = 50;

	// shared empty-probe flag (forward pass). set by any sibling CREATE_FILTER / PROBE_FILTER
	// targeting the same probe table when it detects the probe will be empty.
	// read lock-free (relaxed) in Sink to short-circuit BF build.
	shared_ptr<std::atomic<bool>> probe_empty_flag;
};

class CreateFilterLocalSourceState : public LocalSourceState {
public:
	CreateFilterLocalSourceState() {
		local_current_chunk_id = 0;
		initial = true;
	}

public:
	idx_t local_current_chunk_id;
	idx_t local_partition_id;
	idx_t chunk_from;
	idx_t chunk_to;
	bool initial;
};

class CreateFilterGlobalSourceState : public GlobalSourceState {
public:
	CreateFilterGlobalSourceState(ClientContext &context, const PhysicalCreateFilter &op);

	idx_t MaxThreads() override;

	ClientContext &context;
	ColumnDataScanState scan_state;
	vector<pair<idx_t, idx_t>> chunks_todo;
	std::atomic<idx_t> partition_id;
	vector<shared_ptr<PTBloomFilter>> bloom_filters;
	mutex bf_lock;
};

} // namespace duckdb
