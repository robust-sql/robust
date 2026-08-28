#pragma once

#include "duckdb.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "../optimizer/graph_manager.hpp"
#include "bloom_filter.hpp"

namespace duckdb {

struct ProbeFilterStats;
class PhysicalCreateFilter;

class PhysicalProbeFilterState : public CachingOperatorState {
public:
	PhysicalProbeFilterState() : bloom_filters_initialized(false), bit_vector((STANDARD_VECTOR_SIZE + 7) / 8) {
	}

	vector<shared_ptr<PTBloomFilter>> bloom_filters;
	bool bloom_filters_initialized;

	// reusable buffers to avoid per-chunk heap allocations
	vector<SelectionVector> sel_vector;
	vector<uint8_t> bit_vector;
};

class PhysicalProbeFilter : public CachingPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalProbeFilter(PhysicalPlan &physical_plan, shared_ptr<FilterOperation> filter_operation,
	                    vector<LogicalType> types, idx_t estimated_cardinality, vector<idx_t> bound_column_indices);

	// required virtual methods
	~PhysicalProbeFilter() override = default;

	string GetName() const override;
	string ToString(ExplainFormat format = ExplainFormat::DEFAULT) const override;

	// populate info in query plan
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// state management
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

	bool ParallelOperator() const override {
		return true;
	}

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

protected:
	// operator interface - using ExecuteInternal for CachingPhysicalOperator
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override;

public:
	shared_ptr<FilterOperation> filter_operation;
	bool is_passthrough = false;

	// maps the column indices to resolved chunk column positions
	vector<idx_t> bound_column_indices;

	// references to related CREATE_FILTER operators
	vector<PhysicalCreateFilter *> related_create_filter_vec;
	mutable PhysicalCreateFilter *related_create_filter = nullptr;

	// profiling
	mutable shared_ptr<ProbeFilterStats> profiling_stats;
	mutable bool profiling_checked = false;
};

} // namespace duckdb
