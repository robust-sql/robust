#include "redundant_pair_removal.hpp"
#include "robust_optimizer.hpp"
#include "table_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/table_statistics.hpp"
#include "duckdb/common/types/hyperloglog.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "utils/debug_utils.hpp"
#include <algorithm>

namespace duckdb {

static FilterOperation *FindOperation(unordered_map<LogicalOperator *, vector<FilterOperation>> &filter_ops,
                                      idx_t sequence_number) {
	for (auto &entry : filter_ops) {
		auto &table_op = entry.first;
		auto &ops = entry.second;
		for (auto &op : ops) {
			if (op.sequence_number == sequence_number) {
				return &op;
			}
		}
	}

	return nullptr;
}

static void EraseOperation(unordered_map<LogicalOperator *, vector<FilterOperation>> &filter_ops,
                           idx_t sequence_number) {
	for (auto map_it = filter_ops.begin(); map_it != filter_ops.end(); map_it++) {
		auto &ops = map_it->second;
		for (auto op_it = ops.begin(); op_it != ops.end(); op_it++) {
			if (op_it->sequence_number != sequence_number) {
				continue;
			}
			ops.erase(op_it);
			if (ops.empty()) {
				filter_ops.erase(map_it);
			}
			return;
		}
	}
}

// remove selected pair
static void RemoveFilterPair(const FilterOpPair &pair,
                             unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
                             unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops) {
	auto &filter_ops = pair.create_op.is_forward_pass ? forward_filter_ops : backward_filter_ops;

	if (pair.create_op.is_forward_pass) {
		EraseOperation(filter_ops, pair.create_op.sequence_number);
		EraseOperation(filter_ops, pair.probe_op.sequence_number);
		return;
	}

	// if backward pass, CREATE needs to be partially deleted
	auto *create_op = FindOperation(filter_ops, pair.create_op.sequence_number);
	if (!create_op) {
		// a broadcast CREATE shared with earlier pairs is already gone; drop the PROBE alone
		EraseOperation(filter_ops, pair.probe_op.sequence_number);
		return;
	}
	for (const auto &target_probe : pair.probe_op.probe_columns) {
		for (idx_t i = 0; i < create_op->probe_columns.size(); i++) {
			const auto &current_probe = create_op->probe_columns[i];

			if (current_probe.table_index != target_probe.table_index ||
			    current_probe.column_index != target_probe.column_index) {
				continue;
			}

			create_op->build_columns.erase(create_op->build_columns.begin() + (int)i);
			create_op->probe_columns.erase(create_op->probe_columns.begin() + (int)i);
			break;
		}
	}

	EraseOperation(filter_ops, pair.probe_op.sequence_number);

	// if CREATE is empty, delete it
	if (create_op->probe_columns.empty()) {
		EraseOperation(filter_ops, pair.create_op.sequence_number);
	}
}

static unique_ptr<BaseStatistics> GetColumnStatistics(RobustOptimizerContextState &state,
                                                      const ColumnBinding &binding) {
	auto resolved_binding = state.ResolveColumnBinding(binding);
	auto table_it = state.table_mgr.table_lookup.find(resolved_binding.table_index);
	if (table_it == state.table_mgr.table_lookup.end()) {
		return nullptr;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get || resolved_binding.column_index >= get->GetColumnIds().size()) {
		D_PRINTF("No matching base table found");
		return nullptr;
	}
	auto &column_id = get->GetColumnIds()[resolved_binding.column_index];

	if (!get->function.statistics && !get->function.statistics_extended) {
		return nullptr;
	}
	unique_ptr<BaseStatistics> column_statistics;
	if (get->function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get->bind_data.get(), column_id);
		column_statistics = get->function.statistics_extended(state.context, input);
	} else {
		D_ASSERT(get->function.statistics);
		column_statistics = get->function.statistics(state.context, get->bind_data.get(), column_id.GetPrimaryIndex());
	}
	return column_statistics;
}

static idx_t GetBaseTableRowCount(RobustOptimizerContextState &state, const ColumnBinding &binding) {
	auto resolved_binding = state.ResolveColumnBinding(binding);
	auto table_it = state.table_mgr.table_lookup.find(resolved_binding.table_index);
	if (table_it == state.table_mgr.table_lookup.end()) {
		return DConstants::INVALID_INDEX;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get || !get->function.cardinality) {
		return DConstants::INVALID_INDEX;
	}
	auto table_stats = get->function.cardinality(state.context, get->bind_data.get());
	if (!table_stats || !table_stats->has_estimated_cardinality) {
		return DConstants::INVALID_INDEX;
	}
	return table_stats->estimated_cardinality;
}

static unique_ptr<HyperLogLog> GetColumnHLL(RobustOptimizerContextState &state, const ColumnBinding &binding) {
	auto resolved_binding = state.ResolveColumnBinding(binding);
	auto table_it = state.table_mgr.table_lookup.find(resolved_binding.table_index);
	if (table_it == state.table_mgr.table_lookup.end()) {
		return nullptr;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get) {
		D_PRINTF("No matching base table found");
		return nullptr;
	}
	auto table = get->GetTable();
	if (!table || !table->IsDuckTable()) {
		return nullptr;
	}
	const auto &column_id = get->GetColumnIds()[resolved_binding.column_index];
	StorageIndex storage_index;
	if (!get->TryGetStorageIndex(column_id, storage_index)) {
		return nullptr;
	}
	auto &duck_table = table->Cast<DuckTableEntry>();
	TableStatistics table_statistics;
	duck_table.GetStorage().GetRowGroupCollection()->CopyStats(table_statistics);
	auto stats_lock = table_statistics.GetLock();
	auto &column_statistics = table_statistics.GetStats(*stats_lock, storage_index.GetPrimaryIndex());
	if (!column_statistics.HasDistinctStats()) {
		return nullptr;
	}
	auto &distinct_statistics = column_statistics.DistinctStats();
	if (!distinct_statistics.log || distinct_statistics.sample_count.load() == 0 ||
	    distinct_statistics.total_count.load() == 0) {
		return nullptr;
	}
	return distinct_statistics.log->Copy();
}

// check HLL dominance as approximate evidence as whether probe side key set is contained in build side key set
static bool HLLDominates(RobustOptimizerContextState &state, const ColumnBinding &build_binding,
                         const ColumnBinding &probe_binding) {
	auto build_hll = GetColumnHLL(state, build_binding);
	auto probe_hll = GetColumnHLL(state, probe_binding);
	if (!build_hll || !probe_hll) {
		return false;
	}

	for (idx_t i = 0; i < HyperLogLog::M; i++) {
		if (build_hll->GetRegister(i) < probe_hll->GetRegister(i)) {
			return false;
		}
	}
	return true;
}

enum class IncomingFilterStatus { NONE, ONE, MULTIPLE };

struct FilterTableState {
	IncomingFilterStatus status = IncomingFilterStatus::NONE;
	idx_t source_idx = DConstants::INVALID_INDEX;
};

static void TightenProbeRange(BaseStatistics &stats, const TableFilter &filter) {
	if (filter.filter_type == TableFilterType::CONJUNCTION_AND) {
		for (auto &child : filter.Cast<ConjunctionAndFilter>().child_filters) {
			TightenProbeRange(stats, *child);
		}
		return;
	}
	if (filter.filter_type != TableFilterType::CONSTANT_COMPARISON) {
		return;
	}

	auto &f = filter.Cast<ConstantFilter>();
	auto lo = NumericStats::Min(stats);
	auto hi = NumericStats::Max(stats);

	switch (f.comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		lo = std::max(lo, f.constant);
		hi = std::min(hi, f.constant);
		break;

	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		lo = std::max(lo, f.constant);
		break;

	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		hi = std::min(hi, f.constant);
		break;

	default:
		return;
	}

	if (lo <= hi) {
		NumericStats::SetMin(stats, lo);
		NumericStats::SetMax(stats, hi);
	}
}

static void TightenWithLocalPredicate(RobustOptimizerContextState &state, const ColumnBinding &binding,
                                      BaseStatistics &stats) {
	if (!stats.GetType().IsIntegral() || !NumericStats::HasMinMax(stats)) {
		return;
	}
	auto resolved = state.ResolveColumnBinding(binding);
	auto table_it = state.table_mgr.table_lookup.find(resolved.table_index);
	if (table_it == state.table_mgr.table_lookup.end()) {
		return;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get || resolved.column_index >= get->GetColumnIds().size()) {
		D_PRINTF("No matching base table found");
		return;
	}

	auto col = get->GetColumnIds()[resolved.column_index].GetPrimaryIndex();
	auto it = get->table_filters.filters.find(col);
	if (it != get->table_filters.filters.end()) {
		TightenProbeRange(stats, *it->second);
	}
}

static bool HasFilteringLocalPredicate(RobustOptimizerContextState &state, const FilterOpPair &pair) {
	const auto &pair_op = pair.probe_op;
	const auto &build_columns = pair_op.build_columns;
	const auto &probe_columns = pair_op.probe_columns;
	auto build_table_idx = pair_op.build_table_idx;

	// build_columns[i] pairs with probe_columns[i]; a broadcast CREATE can carry more build
	// columns than this edge has probe columns, and we cannot pair them up. treat as filtered
	if (build_columns.size() != probe_columns.size()) {
		return true;
	}

	if (state.table_with_filters.count(build_table_idx)) {
		return true;
	}

	auto table_it = state.table_mgr.table_lookup.find(build_table_idx);
	if (table_it == state.table_mgr.table_lookup.end()) {
		return true;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get) {
		D_PRINTF("No matching base table found");
		return true;
	}
	auto &column_ids = get->GetColumnIds();

	for (auto &entry : get->table_filters.filters) {
		auto &filter = *entry.second;

		if (filter.filter_type == TableFilterType::OPTIONAL_FILTER) {
			continue;
		}

		idx_t filter_column_index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == entry.first) {
				filter_column_index = i;
				break;
			}
		}
		if (filter_column_index == DConstants::INVALID_INDEX) {
			return true;
		}

		// check if this local predicate is non-selective
		ColumnBinding filter_binding(build_table_idx, filter_column_index);
		auto build_stats = GetColumnStatistics(state, filter_binding);
		if (!build_stats) {
			return true;
		}
		auto build_result = filter.CheckStatistics(*build_stats);
		if (build_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			continue;
		}

		// check if this local predicate is on join key
		auto resolved_filter_binding = state.ResolveColumnBinding(filter_binding);
		idx_t key_index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < build_columns.size(); i++) {
			if (state.ResolveColumnBinding(build_columns[i]) == resolved_filter_binding) {
				key_index = i;
				break;
			}
		}
		if (key_index == DConstants::INVALID_INDEX) {
			return true;
		}
		auto probe_binding = state.ResolveColumnBinding(probe_columns[key_index]);
		auto probe_stats = GetColumnStatistics(state, probe_binding);
		if (!probe_stats) {
			return true;
		}

		TightenWithLocalPredicate(state, probe_binding, *probe_stats);

		auto probe_result = filter.CheckStatistics(*probe_stats);
		if (probe_result != FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return true;
		}
	}

	return false;
}

static bool IsRedundant(RobustOptimizerContextState &state, const FilterOpPair &pair) {
	const auto &build_columns = pair.probe_op.build_columns;
	const auto &probe_columns = pair.probe_op.probe_columns;

	// see HasFilteringLocalPredicate: unpairable columns mean we cannot reason about the pair
	if (build_columns.empty() || build_columns.size() != probe_columns.size()) {
		return false;
	}

	// every column carries its own bloom filter and removal drops all of them, so the pair is
	// redundant only if none of them can reject anything
	for (idx_t i = 0; i < build_columns.size(); ++i) {
		auto build_stats = GetColumnStatistics(state, build_columns[i]);
		auto probe_stats = GetColumnStatistics(state, probe_columns[i]);
		if (!build_stats || !probe_stats) {
			return false;
		}
		if (!build_stats->GetType().IsIntegral() || !probe_stats->GetType().IsIntegral()) {
			return false;
		}
		if (!NumericStats::HasMinMax(*build_stats) || !NumericStats::HasMinMax(*probe_stats)) {
			return false;
		}

		TightenWithLocalPredicate(state, probe_columns[i], *probe_stats);

		auto build_min = NumericStats::Min(*build_stats).GetValue<int64_t>();
		auto build_max = NumericStats::Max(*build_stats).GetValue<int64_t>();
		auto probe_min = NumericStats::Min(*probe_stats).GetValue<int64_t>();
		auto probe_max = NumericStats::Max(*probe_stats).GetValue<int64_t>();

		// if probe range is outside of build range, then not redundant
		if (probe_min < build_min || probe_max > build_max) {
			return false;
		}
		auto build_range = build_max - build_min + 1;
		auto build_row_count = GetBaseTableRowCount(state, build_columns[i]);
		bool redundant_by_row_count =
		    build_row_count != DConstants::INVALID_INDEX && build_row_count == static_cast<idx_t>(build_range);
		if (redundant_by_row_count) {
			continue;
		}
		if (!HLLDominates(state, build_columns[i], probe_columns[i])) {
			return false;
		}
	}
	return true;
}

void RemoveRedundantPairs(RobustOptimizerContextState &state, vector<FilterOpPair> &filter_pairs,
                          unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
                          unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops) {
	map<idx_t, FilterTableState> table_states;

	// initialzize status of tables using local predicates
	for (const auto &entry : state.table_mgr.table_lookup) {
		table_states[entry.first] = FilterTableState();
	}
	vector<bool> remove_pair(filter_pairs.size(), false);

	// iterate all pairs in sequence to mark removable pairs
	for (idx_t pair_idx = 0; pair_idx < filter_pairs.size(); pair_idx++) {
		const auto &pair = filter_pairs[pair_idx];
		const auto &pair_op = pair.probe_op;
		auto &build_state = table_states[pair_op.build_table_idx];
		auto &probe_state = table_states[pair_op.probe_table_idx];
		bool redundant = false;

		if (build_state.status == IncomingFilterStatus::NONE && !HasFilteringLocalPredicate(state, pair) &&
		    IsRedundant(state, pair)) {
			redundant = true;
		}

		// peel from turnaround
		if (build_state.status == IncomingFilterStatus::ONE && build_state.source_idx == pair_op.probe_table_idx &&
		    !HasFilteringLocalPredicate(state, pair) && IsRedundant(state, pair)) {
			redundant = true;
		}

		// if a pair is marked as redundant, it will not be potentially filtering source for the next pair
		if (redundant) {
			remove_pair[pair_idx] = true;
			continue;
		}

		if (probe_state.status == IncomingFilterStatus::NONE) {
			probe_state.status = IncomingFilterStatus::ONE;
		} else if (probe_state.status == IncomingFilterStatus::ONE) {
			probe_state.status = IncomingFilterStatus::MULTIPLE;
		}
		probe_state.source_idx = pair_op.build_table_idx;
	}

	for (idx_t pair_idx = 0; pair_idx < filter_pairs.size(); pair_idx++) {
		if (!remove_pair[pair_idx]) {
			continue;
		}

		auto &pair = filter_pairs[pair_idx];
		RemoveFilterPair(pair, forward_filter_ops, backward_filter_ops);
	}
}

} // namespace duckdb
