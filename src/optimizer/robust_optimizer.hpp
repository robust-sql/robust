#pragma once

#include "graph_manager.hpp"
#include "table_manager.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/common/types/hyperloglog.hpp"

namespace duckdb {

// tree node for rooted MST representation (used by Robust DAG)
struct TreeNode {
	idx_t table_idx;
	LogicalOperator *table_op;
	vector<TreeNode *> children;
	TreeNode *parent;
	int level;                // distance from root (root = 0)
	JoinEdge *edge_to_parent; // null for root

	TreeNode(idx_t idx, LogicalOperator *op)
	    : table_idx(idx), table_op(op), parent(nullptr), level(0), edge_to_parent(nullptr) {
	}
};

// edge in physical plan DAG (stores resolved column bindings for label)
struct PhysicalDAGEdge {
	idx_t parent_table;
	idx_t child_table;
	vector<ColumnBinding> parent_cols;
	vector<ColumnBinding> child_cols;
};

// node in physical plan DAG (supports multiple parents for multi-way joins)
struct PhysicalDAGNode {
	idx_t table_idx;
	LogicalOperator *table_op;
	vector<PhysicalDAGNode *> children;
	vector<PhysicalDAGNode *> parents;
	vector<PhysicalDAGEdge> edges_to_parents; // one per parent, same ordering
	int level;

	PhysicalDAGNode(idx_t idx, LogicalOperator *op) : table_idx(idx), table_op(op), level(0) {
	}
};

struct PhysicalDAGComponent {
	vector<PhysicalDAGNode *> comp_nodes;
	vector<PhysicalDAGNode *> roots;
	PhysicalDAGNode *anchor;
};

// column key for equivalence class union-find
using ColKey = std::pair<idx_t, idx_t>; // (table_idx, column_idx)

class RobustOptimizerContextState : public ClientContextState {
public:
	explicit RobustOptimizerContextState(ClientContext &ctx) : context(ctx) {
	}

	ClientContext &context;

	vector<JoinEdge> join_edges;
	//	map<table_id, idx_t> table_cardinalities;
	map<LogicalOperator *, idx_t> operator_to_table_id;

	TableManager table_mgr;
	vector<LogicalOperator *> join_ops;
	vector<JoinEdge> mst_edges;

	unordered_map<ColumnBinding, ColumnBinding, ColumnBindingHashFunction> rename_col_bindings;

	unordered_set<idx_t> table_with_filters;

	bool exist_cycle = false;

public:
	// extract all the join edges from the plan
	// vector<JoinEdge> ExtractOperators(LogicalOperator &plan, vector<LogicalOperator*> &join_ops);
	vector<JoinEdge> ExtractOperators(LogicalOperator &plan);
	void ExtractOperatorsRecursive(LogicalOperator &plan, vector<LogicalOperator *> &join_ops);
	map<table_id, TableInfo> get_value();
	vector<JoinEdge> CreateJoinEdges(vector<LogicalOperator *> &join_ops);
	vector<JoinEdge> LargestRoot(vector<JoinEdge> &edges);

	// build rooted tree from MST edges with largest table as root
	TreeNode *BuildRootedTree(vector<JoinEdge> &mst_edges) const;

	// void CreateForwardPassModifications(LogicalOperator *smaller_table_op, LogicalOperator *larger_table_op,
	// 														const vector<ColumnBinding> &smaller_columns, const
	// vector<ColumnBinding>
	// &larger_columns, 														unordered_map<LogicalOperator*,
	// unique_ptr<LogicalOperator>> &forward_pass);
	//
	// void CreateBackwardPassModifications(LogicalOperator *smaller_table_op, LogicalOperator *larger_table_op,
	// 														const vector<ColumnBinding> &smaller_columns, const
	// vector<ColumnBinding>
	// &larger_columns, 														unordered_map<LogicalOperator*,
	// unique_ptr<LogicalOperator>> &backward_pass);
	//
	std::pair<unordered_map<LogicalOperator *, vector<FilterOperation>>,
	          unordered_map<LogicalOperator *, vector<FilterOperation>>>
	GenerateStageModifications(const vector<JoinEdge> &mst_edges);

	std::pair<unordered_map<LogicalOperator *, vector<FilterOperation>>,
	          unordered_map<LogicalOperator *, vector<FilterOperation>>>
	GenerateStageModificationsFromDAG(vector<PhysicalDAGNode *> &all_nodes, map<ColKey, ColKey> &uf_paren,
	                                  vector<FilterOpPair> &filter_pairs);

	unique_ptr<LogicalOperator> BuildStackedBFOperators(unique_ptr<LogicalOperator> base_plan,
	                                                    const vector<FilterOperation> &filter_ops,
	                                                    bool reverse_order = false);

	unique_ptr<LogicalOperator>
	ApplyStageModifications(unique_ptr<LogicalOperator> plan,
	                        const unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
	                        const unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops);

	// helper to link PROBE_FILTER operators to their corresponding CREATE_FILTER operators
	void LinkProbeFilterToCreateFilter(LogicalOperator *plan);

	// set up dynamic filter pushdown for forward-pass CREATE_FILTER operators
	void SetupDynamicFilterPushdown(LogicalOperator *plan);

	// pass 1: lift BF operator block above MARK_JOIN (probe chain → above MARK_JOIN)
	void LiftCreateFilterAboveMarkJoin(unique_ptr<LogicalOperator> &plan);

	// pass 2: lift BF operator block above FILTER (handles all FILTER cases)
	void LiftCreateFilterAboveFilter(unique_ptr<LogicalOperator> &plan);

	unique_ptr<BaseStatistics> GetColumnStatistics(const ColumnBinding &binding);

	idx_t GetBaseTableRowCount(const ColumnBinding &binding);

	unique_ptr<HyperLogLog> GetColumnHLL(const ColumnBinding &binding);

	bool HLLDominates(const ColumnBinding &build_binding, const ColumnBinding &probe_binding);

	void TightenWithLocalPredicate(const ColumnBinding &binding, BaseStatistics &stats);

	bool HasFilteringLocalPredicate(const FilterOpPair &pair);

	bool IsRedundant(const FilterOpPair &pair);

	void RemoveRedundantPairs(vector<FilterOpPair> &filter_pairs,
	                          unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
	                          unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops);

	// resolve column binding through rename chain to get base table binding
	ColumnBinding ResolveColumnBinding(const ColumnBinding &binding) const;

	// debug functions
	void DebugPrintGraph(const vector<JoinEdge> &edges) const;
	void DebugPrintMST(const vector<JoinEdge> &mst_edges, const vector<FilterOperation> &filter_operations);

	// print DAG as ASCII tree (gated by robust_display_dag setting)
	void PrintDAG(TreeNode *root);

	// build and print DAG from DuckDB's join order (gated by robust_display_physical_dag)
	vector<PhysicalDAGNode *> BuildPhysicalPlanDAG(LogicalOperator *op, map<ColKey, ColKey> &uf_parent);
	void PrintPhysicalPlanDAG(LogicalOperator *op);

	// flip non-largest roots to leaves in the DAG
	void FlipRootsToLeaves(vector<PhysicalDAGNode *> &all_nodes);

	unique_ptr<LogicalOperator> PreOptimize(unique_ptr<LogicalOperator> plan);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> plan);

	// entry point for extension framework
	static void PreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

} // namespace duckdb
