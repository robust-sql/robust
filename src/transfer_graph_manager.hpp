#pragma once

#include "table_operator_manager.hpp"
#include "dag.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/vector.hpp"

#include <iostream>

namespace duckdb {

class EdgeInfo {
public:
	EdgeInfo(const LogicalType &type, LogicalOperator &left, const ColumnBinding &left_binding, LogicalOperator &right,
	         const ColumnBinding &right_binding)
	    : return_type(type), left_table(left), left_binding(left_binding), right_table(right),
	      right_binding(right_binding) {
	}

	LogicalType return_type;

	LogicalOperator &left_table;
	ColumnBinding left_binding;

	LogicalOperator &right_table;
	ColumnBinding right_binding;

	bool protect_left = false;
	bool protect_right = false;
};

// Join Key Table Groups
struct JoinKeyTableGroup {
	LogicalType return_type;
	unordered_set<idx_t> table_ids;
	idx_t leader_id;
	ColumnBinding leader_column_binding;

	JoinKeyTableGroup(const LogicalType &return_type, const idx_t table_id)
	    : return_type(return_type), leader_id(std::numeric_limits<idx_t>::max()) {
		table_ids.insert(table_id);
	}

	void Union(const JoinKeyTableGroup &other) {
		D_ASSERT(return_type == other.return_type);
		table_ids.insert(other.table_ids.begin(), other.table_ids.end());
	}

	void RegisterLeader(const idx_t table_id, const ColumnBinding &column_binding) {
		if (std::numeric_limits<idx_t>::max() == leader_id) {
			leader_id = table_id;
			leader_column_binding = column_binding;
		}
	}

	void Print() const {
		std::cout << "[Leader: " << leader_id << "]\t";
		for (auto &id : table_ids) {
			std::cout << id << " ";
		}
		std::cout << "\n";
	}
};

struct ColumnBindingHashFunc {
	size_t operator()(const ColumnBinding &key) const {
		return std::hash<uint64_t> {}(key.table_index.index) ^ (std::hash<uint64_t> {}(key.column_index) << 1);
	}
};

using JoinKeySet = unordered_set<ColumnBinding, ColumnBindingHashFunc>;
using TableJoinKeyMap = unordered_map<idx_t, JoinKeySet>;
using BindingParentMap = unordered_map<ColumnBinding, ColumnBinding, ColumnBindingHashFunc>;
using BindingGroupMap = unordered_map<ColumnBinding, shared_ptr<JoinKeyTableGroup>, ColumnBindingHashFunc>;

class TransferGraphManager {
public:
	explicit TransferGraphManager(ClientContext &context) : context(context), table_operator_manager(context) {
	}

	ClientContext &context;
	TableOperatorManager table_operator_manager;
	TransferGraph transfer_graph;
	vector<LogicalOperator *> transfer_order;

public:
	bool Build(LogicalOperator &op);
	void AddFilterPlan(idx_t create_table, const shared_ptr<FilterPlan> &filter_plan, bool reverse);
	void PrintTransferPlan();

private:
	void ExtractEdgesInfo(const vector<reference<LogicalOperator>> &join_operators);
	void CreateOriginTransferPlan();
	void CreateTransferPlanUpdated();
	void LargestRoot(vector<LogicalOperator *> &sorted_nodes);
	void LargestRootUpdated(vector<LogicalOperator *> &sorted_nodes);

	pair<idx_t, idx_t> FindEdge(const unordered_set<idx_t> &constructed_set,
	                            const unordered_set<idx_t> &unconstructed_set);

private:
	//! Classify all tables into three categories: intermediate table, unfiltered table, and filtered table.
	void ClassifyTables();
	void SkipUnfilteredTable(const vector<reference<LogicalOperator>> &joins);

	//! From table id to its join keys
	TableJoinKeyMap table_join_keys;
	//! From join keys to its table groups
	BindingGroupMap table_groups;
	//! Table categories
	unordered_set<idx_t> unfiltered_table;
	unordered_set<idx_t> filtered_table;
	unordered_set<idx_t> intermediate_table;

private:
	unordered_map<idx_t, unordered_map<idx_t, shared_ptr<EdgeInfo>>> neighbor_matrix;
	vector<shared_ptr<EdgeInfo>> selected_edges;
};
} // namespace duckdb
