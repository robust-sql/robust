#include "robust_optimizer.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/types.hpp"
#include "table_manager.hpp"
#include "graph_manager.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/set.hpp"
#include <algorithm>
#include "duckdb/common/vector.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "../operators/logical_create_filter.hpp"
#include "../operators/logical_probe_filter.hpp"
#include "utils/debug_utils.hpp"
#include "robust_profiling.hpp"
#include "../utils/dag_printer.hpp"
#include <chrono>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/table_statistics.hpp"
#include "duckdb/common/types/hyperloglog.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"

namespace duckdb {
// class LogicalCreateFilter;
// class LogicalProbeFilter;

vector<JoinEdge> RobustOptimizerContextState::ExtractOperators(LogicalOperator &plan) {
	vector<LogicalOperator *> join_ops;
	vector<TableInfo> table_infos;

	// pass 1: collect the base tables and join operators
	ExtractOperatorsRecursive(plan, join_ops);

	// debug: print summary of registered nodes
	D_PRINT("\n=== REGISTERED NODES SUMMARY ===");
	for (const auto &entry : table_mgr.table_lookup) {
		D_PRINTF("  table_idx=%llu (type=%d, cardinality=%llu)", (unsigned long long)entry.first,
		         (int)entry.second.table_op->type, (unsigned long long)entry.second.estimated_cardinality);
	}
	D_PRINTF("Total registered nodes: %zu", table_mgr.table_lookup.size());
	D_PRINTF("Total join operators found: %zu\n", join_ops.size());

	// pass 2: create JoinEdges with table information
	return CreateJoinEdges(join_ops);
}

void RobustOptimizerContextState::ExtractOperatorsRecursive(LogicalOperator &plan,
                                                            vector<LogicalOperator *> &join_ops) {
	LogicalOperator *op = &plan;

	// step 1: collect all join operators
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		LogicalComparisonJoin &join = op->Cast<LogicalComparisonJoin>();
		switch (join.join_type) {
		case JoinType::INNER:
		case JoinType::LEFT:
		case JoinType::RIGHT:
		case JoinType::SEMI:
		case JoinType::RIGHT_SEMI: {
			if (std::any_of(join.conditions.begin(), join.conditions.end(), [](const JoinCondition &jc) {
				    return jc.comparison == ExpressionType::COMPARE_EQUAL &&
				           jc.left->type == ExpressionType::BOUND_COLUMN_REF &&
				           jc.right->type == ExpressionType::BOUND_COLUMN_REF;
			    })) {
				// JoinEdge edge(join);
				join_ops.push_back(op);
				break;
			}
		}
		default:
			break;
		}
	}

	switch (op->type) {
	case LogicalOperatorType::LOGICAL_FILTER: {
		auto &logical_filter = op->Cast<LogicalFilter>();
		LogicalOperator *child = op->children[0].get();

		// record base tables that have LOGICAL_FILTERs
		if (!logical_filter.expressions.empty()) {
			auto *current = child;
			while (true) {
				if (current->children.size() > 2 || current->children.empty()) {
					break;
				} else if (current->children.size() == 2) {
					if (current->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
						break;
					}
					auto &join = current->Cast<LogicalComparisonJoin>();
					if (join.join_type != JoinType::MARK) {
						break;
					}
				} else if (current->children.size() == 1) {
					if (current->type == LogicalOperatorType::LOGICAL_GET) {
						break;
					}
				}
				current = current->children[0].get();
			}
			if (current->type == LogicalOperatorType::LOGICAL_GET) {
				auto table_idx = current->Cast<LogicalGet>().table_index;
				table_with_filters.insert(table_idx);
			}
		}

		if (child->type == LogicalOperatorType::LOGICAL_GET) {
			table_mgr.AddTableOperator(child);
			return;
		}

		ExtractOperatorsRecursive(*child, join_ops);
		return;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &agg = op->Cast<LogicalAggregate>();
		if (agg.groups.empty() && agg.grouping_sets.size() <= 1) {
			table_mgr.AddTableOperator(op);
			ExtractOperatorsRecursive(*op->children[0], join_ops);
		} else {
			auto old_refs = agg.GetColumnBindings();
			for (size_t i = 0; i < agg.groups.size(); i++) {
				if (agg.groups[i]->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = agg.groups[i]->Cast<BoundColumnRefExpression>();
					rename_col_bindings.insert({old_refs[i], col_ref.binding});
				}
			}
			ExtractOperatorsRecursive(*op->children[0], join_ops);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto old_refs = op->GetColumnBindings();
		for (size_t i = 0; i < op->expressions.size(); i++) {
			if (op->expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = op->expressions[i]->Cast<BoundColumnRefExpression>();
				rename_col_bindings.insert({old_refs[i], col_ref.binding});
			} else if (op->expressions[i]->type == ExpressionType::BOUND_FUNCTION) {
				auto &func_ref = op->expressions[i]->Cast<BoundFunctionExpression>();
				for (const auto &arg : func_ref.children) {
					if (arg->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = arg->Cast<BoundColumnRefExpression>();
						rename_col_bindings.insert({old_refs[i], col_ref.binding});
					}
				}
			}
		}
		ExtractOperatorsRecursive(*op->children[0], join_ops);
		return;
	}
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT: {
		table_mgr.AddTableOperator(op);
		ExtractOperatorsRecursive(*op->children[0], join_ops);
		ExtractOperatorsRecursive(*op->children[1], join_ops);
		return;
	}
	case LogicalOperatorType::LOGICAL_WINDOW: {
		table_mgr.AddTableOperator(op);
		ExtractOperatorsRecursive(*op->children[0], join_ops);
		return;
	}
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_CTE_REF:
		D_PRINTF("[NODE_REG] Registering base table scan, type=%d", (int)op->type);
		table_mgr.AddTableOperator(op);
		return;
	default:
		for (auto &child : op->children) {
			ExtractOperatorsRecursive(*child, join_ops);
		}
	}
}

ColumnBinding RobustOptimizerContextState::ResolveColumnBinding(const ColumnBinding &binding) const {
	ColumnBinding current = binding;
	set<pair<idx_t, idx_t>> visited;

	// follow the rename chain until we find a base table binding
	while (true) {
		auto key = make_pair(current.table_index, current.column_index);
		if (visited.count(key)) {
			D_PRINTF("WARNING: Cycle detected in rename_col_bindings for binding (%llu.%llu)",
			         (unsigned long long)current.table_index, (unsigned long long)current.column_index);
			break;
		}
		visited.insert(key);

		// check if this binding exists in the rename map
		auto it = rename_col_bindings.find(current);
		if (it != rename_col_bindings.end()) {
			current = it->second;
		} else {
			// no more renames, this is the base binding
			break;
		}
	}

	return current;
}

vector<JoinEdge> RobustOptimizerContextState::CreateJoinEdges(vector<LogicalOperator *> &join_ops) {
	vector<JoinEdge> edges;
	for (auto &op : join_ops) {
		auto &join = op->Cast<LogicalComparisonJoin>();

		vector<ColumnBinding> left_columns, right_columns;
		vector<ColumnBinding> resolved_left_columns, resolved_right_columns;

		for (const JoinCondition &cond : join.conditions) {
			if (cond.comparison == ExpressionType::COMPARE_EQUAL &&
			    cond.left->type == ExpressionType::BOUND_COLUMN_REF &&
			    cond.right->type == ExpressionType::BOUND_COLUMN_REF) {
				// store original bindings
				ColumnBinding left_binding = cond.left->Cast<BoundColumnRefExpression>().binding;
				ColumnBinding right_binding = cond.right->Cast<BoundColumnRefExpression>().binding;

				left_columns.push_back(left_binding);
				right_columns.push_back(right_binding);

				// resolve bindings through rename chain
				resolved_left_columns.push_back(ResolveColumnBinding(left_binding));
				resolved_right_columns.push_back(ResolveColumnBinding(right_binding));
			}
		}

		if (!left_columns.empty() && !right_columns.empty()) {
			// get table indices from first resolved column
			idx_t left_table_idx = resolved_left_columns[0].table_index;
			idx_t right_table_idx = resolved_right_columns[0].table_index;

			// verify these table indices exist in our table manager
			if (table_mgr.table_lookup.find(left_table_idx) != table_mgr.table_lookup.end() &&
			    table_mgr.table_lookup.find(right_table_idx) != table_mgr.table_lookup.end()) {
				// use resolved column bindings in the JoinEdge so they match child bindings in CreatePlan
				JoinEdge edge(left_table_idx, right_table_idx, resolved_left_columns, resolved_right_columns,
				              resolved_left_columns.size(), join.join_type);
				edges.push_back(edge);
			} else {
				D_PRINTF("WARNING: Resolved table indices (%llu, %llu) not found in table_lookup",
				         (unsigned long long)left_table_idx, (unsigned long long)right_table_idx);
			}
		}
	}

	return edges;
}

vector<JoinEdge> RobustOptimizerContextState::LargestRoot(vector<JoinEdge> &edges) {
	// step 1: find largest table by cardinality
	idx_t largest_table_idx = 0;
	idx_t max_cardinality = 0;
	for (auto &table_info : table_mgr.table_ops) {
		if (table_info.estimated_cardinality > max_cardinality) {
			max_cardinality = table_info.estimated_cardinality;
			largest_table_idx = table_info.table_idx;
		}
	}

	// step 2: build MST (maximum) using Prim's algorithm starting from largest table
	unordered_set<idx_t> mst_nodes;
	vector<JoinEdge> mst_edges;

	mst_nodes.insert(largest_table_idx);

	while (mst_nodes.size() < table_mgr.table_ops.size() && !edges.empty()) {
		const JoinEdge *best_edge = nullptr;
		idx_t max_weight = 0;
		max_cardinality = 0;
		for (JoinEdge &edge : edges) {
			bool left_in_mst = mst_nodes.count(edge.table_a) > 0;
			bool right_in_mst = mst_nodes.count(edge.table_b) > 0;

			if (left_in_mst != right_in_mst) {
				const idx_t weight = edge.weight;

				// safely lookup cardinalities with bounds checking
				auto left_it = table_mgr.table_lookup.find(edge.table_a);
				auto right_it = table_mgr.table_lookup.find(edge.table_b);

				if (left_it == table_mgr.table_lookup.end() || right_it == table_mgr.table_lookup.end()) {
					// printf("WARNING: Table lookup failed for edge %llu <-> %llu\n", edge.table_a, edge.table_b);
					continue;
				}

				idx_t left_cardinality = left_it->second.estimated_cardinality;
				idx_t right_cardinality = right_it->second.estimated_cardinality;
				const idx_t cardinality = std::min(left_cardinality, right_cardinality);

				if (weight > max_weight || (weight == max_weight && cardinality > max_cardinality)) {
					max_weight = weight;
					max_cardinality = cardinality;
					best_edge = &edge;
				}
			}
		}

		if (!best_edge) {
			D_PRINT("Warning - Disconnected components found. MST incomplete.");
			// TODO: Add Assertion
			break;
		}

		mst_edges.push_back(*best_edge);
		mst_nodes.insert(best_edge->table_a);
		mst_nodes.insert(best_edge->table_b);
	}

	return mst_edges;
}

TreeNode *RobustOptimizerContextState::BuildRootedTree(vector<JoinEdge> &mst_edges) const {
	if (mst_edges.empty()) {
		return nullptr;
	}

	if (table_mgr.table_ops.empty()) {
		D_PRINT("ERROR: BuildRootedTree called with empty table_ops");
		return nullptr;
	}

	// step 1: find largest table (root)
	idx_t root_table_idx = 0;
	idx_t max_cardinality = 0;
	bool found_root = false;

	for (const auto &table_info : table_mgr.table_ops) {
		if (table_info.estimated_cardinality > max_cardinality) {
			max_cardinality = table_info.estimated_cardinality;
			root_table_idx = table_info.table_idx;
			found_root = true;
		}
	}

	if (!found_root) {
		D_PRINT("ERROR: No valid root table found");
		return nullptr;
	}

	// step 2: create nodes for all tables
	unordered_map<idx_t, TreeNode *> table_to_node;
	for (const auto &table_info : table_mgr.table_ops) {
		auto *node = new TreeNode(table_info.table_idx, table_info.table_op);
		table_to_node[table_info.table_idx] = node;
	}

	// verify root node was created
	if (table_to_node.find(root_table_idx) == table_to_node.end() || !table_to_node[root_table_idx]) {
		D_PRINTF("ERROR: Failed to create root node for table %llu", (unsigned long long)root_table_idx);
		// cleanup allocated nodes
		for (auto &pair : table_to_node) {
			delete pair.second;
		}
		return nullptr;
	}

	// step 3: build adjacency list from MST edges (undirected)
	unordered_map<idx_t, vector<pair<idx_t, JoinEdge *>>> adjacency;
	for (auto &edge : mst_edges) {
		adjacency[edge.table_a].push_back({edge.table_b, &edge});
		adjacency[edge.table_b].push_back({edge.table_a, &edge});
	}

	// step 4: BFS from root to assign parent-child relationships and levels
	vector<idx_t> queue;
	unordered_set<idx_t> visited;

	queue.push_back(root_table_idx);
	visited.insert(root_table_idx);
	table_to_node[root_table_idx]->level = 0;

	size_t front = 0;
	while (front < queue.size()) {
		idx_t current = queue[front++];

		// check if current node exists
		if (table_to_node.find(current) == table_to_node.end() || !table_to_node[current]) {
			D_PRINTF("ERROR: Node for table %llu not found in table_to_node", (unsigned long long)current);
			continue;
		}

		TreeNode *current_node = table_to_node[current];

		// process all neighbors
		for (pair<idx_t, JoinEdge *> &adj_entry : adjacency[current]) {
			idx_t &neighbor_idx = adj_entry.first;
			JoinEdge *&edge = adj_entry.second;
			if (visited.count(neighbor_idx) == 0) {
				// verify neighbor node exists
				if (table_to_node.find(neighbor_idx) == table_to_node.end() || !table_to_node[neighbor_idx]) {
					D_PRINTF("ERROR: Child node for table %llu not found", (unsigned long long)neighbor_idx);
					continue;
				}

				// neighbor is a child of current
				TreeNode *child_node = table_to_node[neighbor_idx];
				child_node->parent = current_node;
				child_node->level = current_node->level + 1;
				child_node->edge_to_parent = edge;

				current_node->children.push_back(child_node);

				queue.push_back(neighbor_idx);
				visited.insert(neighbor_idx);
			}
		}
	}

	return table_to_node[root_table_idx];
}

void RobustOptimizerContextState::DebugPrintGraph(const vector<JoinEdge> &edges) const {
	(void)edges;
#ifdef DEBUG
	// Debug: Print all tables
	Printer::Print("=== TABLE INFORMATION ===");
	for (const auto &table_info : table_mgr.table_ops) {
		Printer::PrintF("Table %llu: cardinality=%llu", (unsigned long long)table_info.table_idx,
		                (unsigned long long)table_info.estimated_cardinality);
	}

	// Find largest table
	idx_t largest_table_idx = 0;
	idx_t max_cardinality = 0;
	for (auto &table_info : table_mgr.table_ops) {
		if (table_info.estimated_cardinality > max_cardinality) {
			max_cardinality = table_info.estimated_cardinality;
			largest_table_idx = table_info.table_idx;
		}
	}
	Printer::PrintF("Largest table: %llu (cardinality=%llu)\n", (unsigned long long)largest_table_idx,
	                (unsigned long long)max_cardinality);

	// Debug: Print all join edges
	Printer::Print("=== ALL JOIN EDGES ===");
	for (size_t i = 0; i < edges.size(); i++) {
		const auto &edge = edges[i];
		Printer::PrintF("Edge %zu: %llu <-> %llu (weight=%llu, type=%d)", i, (unsigned long long)edge.table_a,
		                (unsigned long long)edge.table_b, (unsigned long long)edge.weight, (int)edge.join_type);

		// Print column bindings
		string cols_a = "  Columns A: ";
		for (const auto &col : edge.join_columns_a) {
			cols_a += "(" + std::to_string(col.table_index) + "." + std::to_string(col.column_index) + ") ";
		}
		Printer::Print(cols_a);

		string cols_b = "  Columns B: ";
		for (const auto &col : edge.join_columns_b) {
			cols_b += "(" + std::to_string(col.table_index) + "." + std::to_string(col.column_index) + ") ";
		}
		Printer::Print(cols_b);
	}
	Printer::Print("");
#endif
}

void RobustOptimizerContextState::DebugPrintMST(const vector<JoinEdge> &mst_edges,
                                                const vector<FilterOperation> &filter_operations) {
	(void)mst_edges;
	(void)filter_operations;
#ifdef DEBUG
	Printer::Print("=== MST EDGES ===");
	for (size_t i = 0; i < mst_edges.size(); i++) {
		const auto &edge = mst_edges[i];
		Printer::PrintF("MST Edge %zu: %llu <-> %llu (weight=%llu)", i, (unsigned long long)edge.table_a,
		                (unsigned long long)edge.table_b, (unsigned long long)edge.weight);
	}
	Printer::Print("");

	Printer::Print("=== BLOOM FILTER OPERATIONS ===");
	for (size_t i = 0; i < filter_operations.size(); i++) {
		const auto &filter_op = filter_operations[i];

		if (filter_op.is_create) {
			// CREATE operation
			Printer::PrintF("Filter Op %zu: CREATE_FILTER on table %llu", i,
			                (unsigned long long)filter_op.build_table_idx);
			string cols = "  Build columns: ";
			for (const auto &col : filter_op.build_columns) {
				cols += "(" + std::to_string(col.table_index) + "." + std::to_string(col.column_index) + ") ";
			}
			Printer::Print(cols);
		} else {
			// USE operation
			Printer::PrintF("Filter Op %zu: PROBE_FILTER on table %llu (using BF from table %llu)", i,
			                (unsigned long long)filter_op.probe_table_idx,
			                (unsigned long long)filter_op.build_table_idx);
			string build_cols = "  Build columns: ";
			for (const auto &col : filter_op.build_columns) {
				build_cols += "(" + std::to_string(col.table_index) + "." + std::to_string(col.column_index) + ") ";
			}
			Printer::Print(build_cols);

			string probe_cols = "  Probe columns: ";
			for (const auto &col : filter_op.probe_columns) {
				probe_cols += "(" + std::to_string(col.table_index) + "." + std::to_string(col.column_index) + ") ";
			}
			Printer::Print(probe_cols);
		}
	}
	Printer::Print("");
#endif
}

void RobustOptimizerContextState::PrintDAG(TreeNode *root) {
	Value val;
	if (!context.TryGetCurrentSetting("robust_display_dag", val) || !val.GetValue<bool>()) {
		return;
	}
	PrintTransferDAG(root, table_mgr);
}

// helper: collect all base table indices in a subtree
static void CollectBaseTableIndices(LogicalOperator *op, TableManager &table_mgr, unordered_set<idx_t> &indices) {
	if (!op) {
		return;
	}
	auto *info = table_mgr.GetTableInfo(op);
	if (info) {
		indices.insert(info->table_idx);
		return;
	}
	for (auto &child : op->children) {
		CollectBaseTableIndices(child.get(), table_mgr, indices);
	}
}

// union-find helpers for column equivalence classes
static ColKey UFFind(map<ColKey, ColKey> &parent, ColKey x) {
	if (parent.find(x) == parent.end()) {
		parent[x] = x;
	}
	while (parent[x] != x) {
		parent[x] = parent[parent[x]];
		x = parent[x];
	}
	return x;
}

static void UFUnion(map<ColKey, ColKey> &parent, ColKey a, ColKey b) {
	a = UFFind(parent, a);
	b = UFFind(parent, b);
	if (a != b) {
		parent[a] = b;
	}
}

// add a child->parent edge, or merge columns into an existing edge (multi-column join)
static void AddOrMergeDAGEdge(PhysicalDAGNode *parent_node, PhysicalDAGNode *child_node,
                              const ColumnBinding &parent_col, const ColumnBinding &child_col) {
	for (auto &edge : child_node->edges_to_parents) {
		if (edge.parent_table == parent_node->table_idx) {
			// multi-column join: append columns, skipping exact duplicates
			for (idx_t i = 0; i < edge.parent_cols.size(); i++) {
				if (edge.parent_cols[i] == parent_col && edge.child_cols[i] == child_col) {
					return;
				}
			}
			edge.parent_cols.push_back(parent_col);
			edge.child_cols.push_back(child_col);
			return;
		}
	}
	child_node->parents.push_back(parent_node);
	parent_node->children.push_back(child_node);

	PhysicalDAGEdge edge;
	edge.parent_table = parent_node->table_idx;
	edge.child_table = child_node->table_idx;
	edge.parent_cols.push_back(parent_col);
	edge.child_cols.push_back(child_col);
	child_node->edges_to_parents.push_back(edge);
}

// recompute node levels: roots (no parents) = 0, others = max(parent levels) + 1
static void RecomputeDAGLevels(vector<PhysicalDAGNode *> &all_nodes) {
	for (auto *node : all_nodes) {
		node->level = node->parents.empty() ? 0 : -1;
	}
	bool changed = true;
	while (changed) {
		changed = false;
		for (auto *node : all_nodes) {
			if (node->parents.empty()) {
				continue;
			}
			int max_parent = -1;
			bool all_set = true;
			for (auto *p : node->parents) {
				if (p->level < 0) {
					all_set = false;
					break;
				}
				max_parent = std::max(max_parent, p->level);
			}
			if (all_set && max_parent >= 0) {
				int new_level = max_parent + 1;
				if (new_level != node->level) {
					node->level = new_level;
					changed = true;
				}
			}
		}
	}
}

// raw equality condition captured during DFS, before equivalence rerouting.
// used afterwards to reconnect tables the rerouting left without any edge.
// table indices are derivable as col_a/col_b .table_index
struct RawDAGCond {
	ColumnBinding col_a;
	ColumnBinding col_b;
};

// dfs_index lookup with a defensive default for unregistered tables
static int DFSIndexOf(const map<idx_t, int> &dfs_index, idx_t table) {
	auto it = dfs_index.find(table);
	return it == dfs_index.end() ? -1 : it->second;
}

// recursive DFS for building physical DAG
// uses build-first traversal so DFS index matches execution order
// (first-executed = lowest index, last-executed = highest index)
static void PhysicalDAGDFS(LogicalOperator *op, TableManager &table_mgr, RobustOptimizerContextState &state,
                           vector<PhysicalDAGNode *> &all_nodes, map<idx_t, PhysicalDAGNode *> &node_map,
                           map<idx_t, int> &dfs_index, map<ColKey, ColKey> &uf_parent, vector<RawDAGCond> &raw_conds) {
	if (!op) {
		return;
	}

	// base case: registered base table
	auto *info = table_mgr.GetTableInfo(op);
	if (info) {
		bool is_leaf =
		    (op->type == LogicalOperatorType::LOGICAL_GET || op->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN ||
		     op->type == LogicalOperatorType::LOGICAL_EXPRESSION_GET ||
		     op->type == LogicalOperatorType::LOGICAL_DELIM_GET ||
		     op->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT ||
		     op->type == LogicalOperatorType::LOGICAL_CHUNK_GET || op->type == LogicalOperatorType::LOGICAL_CTE_REF);
		if (is_leaf) {
			auto *node = new PhysicalDAGNode(info->table_idx, info->table_op);
			all_nodes.push_back(node);
			node_map[info->table_idx] = node;
			dfs_index[info->table_idx] = (int)all_nodes.size() - 1;
			return;
		}
	}

	// join node
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();

		if (join.join_type == JoinType::MARK) {
			PhysicalDAGDFS(op->children[0].get(), table_mgr, state, all_nodes, node_map, dfs_index, uf_parent,
			               raw_conds);
			return;
		}

		// build-first: visit right child (build) first, then left child (probe)
		// this gives execution order: first-executed tables get lowest DFS index
		PhysicalDAGDFS(op->children[1].get(), table_mgr, state, all_nodes, node_map, dfs_index, uf_parent, raw_conds);
		PhysicalDAGDFS(op->children[0].get(), table_mgr, state, all_nodes, node_map, dfs_index, uf_parent, raw_conds);

		// process each join condition
		for (const JoinCondition &cond : join.conditions) {
			if (cond.comparison != ExpressionType::COMPARE_EQUAL) {
				continue;
			}
			if (cond.left->type != ExpressionType::BOUND_COLUMN_REF ||
			    cond.right->type != ExpressionType::BOUND_COLUMN_REF) {
				continue;
			}

			ColumnBinding left_resolved =
			    state.ResolveColumnBinding(cond.left->Cast<BoundColumnRefExpression>().binding);
			ColumnBinding right_resolved =
			    state.ResolveColumnBinding(cond.right->Cast<BoundColumnRefExpression>().binding);

			// add to equivalence classes
			ColKey left_key = {left_resolved.table_index, left_resolved.column_index};
			ColKey right_key = {right_resolved.table_index, right_resolved.column_index};
			UFUnion(uf_parent, left_key, right_key);

			idx_t table_a = left_resolved.table_index;
			idx_t table_b = right_resolved.table_index;
			if (!node_map.count(table_a) || !node_map.count(table_b)) {
				continue;
			}
			if (table_a == table_b) {
				continue;
			}

			int idx_a = DFSIndexOf(dfs_index, table_a);
			int idx_b = DFSIndexOf(dfs_index, table_b);

			// higher DFS index = later execution = parent (closer to root/top)
			idx_t parent_idx, child_idx;
			int child_dfs;
			ColumnBinding parent_col, child_col;
			if (idx_a > idx_b) {
				parent_idx = table_a;
				parent_col = left_resolved;
				child_idx = table_b;
				child_col = right_resolved;
				child_dfs = idx_b;
			} else {
				parent_idx = table_b;
				parent_col = right_resolved;
				child_idx = table_a;
				child_col = left_resolved;
				child_dfs = idx_a;
			}

			// remember the raw condition so orphaned tables can be reconnected later.
			// must stay below the node_map checks above: the orphan post-pass relies
			// on every recorded condition having produced a parent-side edge, which
			// guarantees reconnection always attaches to the main DAG (no islands)
			raw_conds.push_back({left_resolved, right_resolved});

			// equiv resolution on child side: find shallowest equivalent (highest DFS, != parent)
			// this chains edges along execution order within an equivalence class
			ColKey child_root = UFFind(uf_parent, {child_col.table_index, child_col.column_index});
			idx_t best_child = child_idx;
			int best_child_dfs = child_dfs;
			ColumnBinding best_child_col = child_col;

			// UFFind only mutates mapped values of existing keys (path compression),
			// so iterating uf_parent directly is safe
			for (auto &entry : uf_parent) {
				ColKey key = entry.first;
				if (UFFind(uf_parent, key) != child_root) {
					continue;
				}
				idx_t candidate = key.first;
				if (candidate == parent_idx) {
					continue;
				}
				if (!node_map.count(candidate)) {
					continue;
				}
				int candidate_dfs = DFSIndexOf(dfs_index, candidate);
				if (candidate_dfs > best_child_dfs) {
					best_child = candidate;
					best_child_dfs = candidate_dfs;
					best_child_col = ColumnBinding(key.first, key.second);
				}
			}

			AddOrMergeDAGEdge(node_map[parent_idx], node_map[best_child], parent_col, best_child_col);
		}
		return;
	}

	// other operators: recurse into children
	for (auto &child : op->children) {
		PhysicalDAGDFS(child.get(), table_mgr, state, all_nodes, node_map, dfs_index, uf_parent, raw_conds);
	}
}

vector<PhysicalDAGNode *> RobustOptimizerContextState::BuildPhysicalPlanDAG(LogicalOperator *op,
                                                                            map<ColKey, ColKey> &uf_parent) {
	vector<PhysicalDAGNode *> all_nodes;
	map<idx_t, PhysicalDAGNode *> node_map;
	map<idx_t, int> dfs_index;
	vector<RawDAGCond> raw_conds;

	PhysicalDAGDFS(op, table_mgr, *this, all_nodes, node_map, dfs_index, uf_parent, raw_conds);

	// reconnect orphans: equivalence rerouting can collapse a table's only
	// condition onto an existing edge, leaving it with no edges at all.
	// link such tables directly via one of their raw conditions, preferring
	// the counterpart that executes latest (highest DFS index).
	for (auto *node : all_nodes) {
		if (!node->parents.empty() || !node->children.empty()) {
			continue;
		}
		bool found = false;
		int best_dfs = -1;
		idx_t best_other = 0;
		ColumnBinding best_node_col, best_other_col;
		for (auto &rc : raw_conds) {
			// normalize so node_col is always this node's side
			ColumnBinding node_col, other_col;
			if (rc.col_a.table_index == node->table_idx) {
				node_col = rc.col_a;
				other_col = rc.col_b;
			} else if (rc.col_b.table_index == node->table_idx) {
				node_col = rc.col_b;
				other_col = rc.col_a;
			} else {
				continue;
			}
			idx_t other = other_col.table_index;
			if (!node_map.count(other)) {
				continue;
			}
			int other_dfs = DFSIndexOf(dfs_index, other);
			if (other_dfs > best_dfs) {
				found = true;
				best_dfs = other_dfs;
				best_other = other;
				best_node_col = node_col;
				best_other_col = other_col;
			}
		}
		if (!found) {
			continue;
		}
		if (best_dfs > DFSIndexOf(dfs_index, node->table_idx)) {
			AddOrMergeDAGEdge(node_map[best_other], node, best_other_col, best_node_col);
		} else {
			AddOrMergeDAGEdge(node, node_map[best_other], best_node_col, best_other_col);
		}
	}

	RecomputeDAGLevels(all_nodes);

	return all_nodes;
}

void RobustOptimizerContextState::FlipRootsToLeaves(vector<PhysicalDAGNode *> &all_nodes) {
	unordered_map<idx_t, bool> visited;
	vector<PhysicalDAGComponent> components;
	for (auto *node : all_nodes) {
		// step 1: find all roots
		if (visited.count(node->table_idx)) {
			continue;
		}
		vector<PhysicalDAGNode *> comp_nodes;
		vector<PhysicalDAGNode *> roots;
		PhysicalDAGNode *anchor;
		vector<PhysicalDAGNode *> que;
		auto discover = [&](PhysicalDAGNode *candidate) {
			if (visited.count(candidate->table_idx)) {
				return;
			}
			if (candidate->parents.empty()) {
				roots.push_back(candidate);
			}
			visited[candidate->table_idx] = true;
			que.push_back(candidate);
			comp_nodes.push_back(candidate);
		};
		discover(node);
		while (!que.empty()) {
			auto cur = que.back();
			que.pop_back();
			for (auto &child : cur->children) {
				discover(child);
			}
			for (auto &parent : cur->parents) {
				discover(parent);
			}
		}

		// step 2: find anchor root (largest cardinality)
		if (roots.empty()) {
			D_PRINTF("ERROR: No root found for this component");
			continue;
		}
		anchor = roots[0];
		for (auto *r : roots) {
			if (r->table_op->estimated_cardinality > anchor->table_op->estimated_cardinality) {
				anchor = r;
			}
		}
		components.push_back({comp_nodes, roots, anchor});
	}
	for (auto &comp : components) {
		auto &comp_nodes = comp.comp_nodes;
		auto &roots = comp.roots;
		auto *anchor = comp.anchor;
		if (roots.size() <= 1) {
			continue;
		}

		// step 3: repeatedly flip non-anchor roots until only anchor remains
		// flipping a root can expose new roots (e.g. flipping name reveals aka_name)
		bool flipped = true;
		while (flipped) {
			flipped = false;
			for (auto *node : comp_nodes) {
				if (!node->parents.empty() || node == anchor) {
					continue;
				}
				// non-anchor root: reverse all edges to its children
				auto children_copy = node->children;
				for (auto *child : children_copy) {
					PhysicalDAGEdge edge;
					bool found = false;
					for (idx_t i = 0; i < child->parents.size(); i++) {
						if (child->parents[i] == node) {
							edge = child->edges_to_parents[i];
							child->parents.erase(child->parents.begin() + static_cast<std::ptrdiff_t>(i));
							child->edges_to_parents.erase(child->edges_to_parents.begin() +
							                              static_cast<std::ptrdiff_t>(i));
							found = true;
							break;
						}
					}
					if (!found) {
						continue;
					}
					for (idx_t i = 0; i < node->children.size(); i++) {
						if (node->children[i] == child) {
							node->children.erase(node->children.begin() + static_cast<std::ptrdiff_t>(i));
							break;
						}
					}
					child->children.push_back(node);
					node->parents.push_back(child);

					PhysicalDAGEdge reversed;
					reversed.parent_table = edge.child_table;
					reversed.child_table = edge.parent_table;
					reversed.parent_cols = edge.child_cols;
					reversed.child_cols = edge.parent_cols;
					node->edges_to_parents.push_back(reversed);

					// only mark progress on a real reversal; a childless root cannot
					// be flipped and must not keep this loop spinning forever
					flipped = true;
				}
			}
		}
	}

	// step 4: recompute levels
	RecomputeDAGLevels(all_nodes);
}

void RobustOptimizerContextState::PrintPhysicalPlanDAG(LogicalOperator *op) {
	Value val;
	if (!context.TryGetCurrentSetting("robust_display_physical_dag", val) || !val.GetValue<bool>()) {
		return;
	}

	map<ColKey, ColKey> uf_parent;
	auto all_nodes = BuildPhysicalPlanDAG(op, uf_parent);
	if (all_nodes.empty()) {
		return;
	}
	PrintPhysicalDAG(all_nodes, table_mgr);
}

std::pair<unordered_map<LogicalOperator *, vector<FilterOperation>>,
          unordered_map<LogicalOperator *, vector<FilterOperation>>>
RobustOptimizerContextState::GenerateStageModifications(const vector<JoinEdge> &mst_edges) {
	// step 1: build rooted tree from MST.
	// BuildRootedTree stores non-const JoinEdge* internally for adjacency-list layout; does not mutate elements.
	TreeNode *root =
	    BuildRootedTree(const_cast<vector<JoinEdge> &>(mst_edges)); // NOLINT(cppcoreguidelines-pro-type-const-cast)

	// check if tree building failed
	if (!root) {
		D_PRINT("ERROR: BuildRootedTree returned nullptr, returning empty modifications");
		return {{}, {}};
	}

	// display DAG if setting is enabled
	PrintDAG(root);

	// step 2: collect all nodes organized by level
	unordered_map<int, vector<TreeNode *>> nodes_by_level;
	int max_level = 0;

	// BFS to collect nodes by level
	vector<TreeNode *> queue;
	queue.push_back(root);
	size_t front = 0;

	while (front < queue.size()) {
		TreeNode *node = queue[front++];
		if (!node) {
			D_PRINT("ERROR: Null node encountered during BFS");
			continue;
		}

		nodes_by_level[node->level].push_back(node);
		max_level = std::max(max_level, node->level);

		for (TreeNode *child : node->children) {
			if (child) {
				queue.push_back(child);
			} else {
				D_PRINT("ERROR: Null child node encountered");
			}
		}
	}

	unordered_map<LogicalOperator *, vector<FilterOperation>> forward_filter_ops;
	unordered_map<LogicalOperator *, vector<FilterOperation>> backward_filter_ops;

	// sequence counter to preserve operation order
	idx_t sequence = 0;

	// sort nodes at each level by cardinality ascending so PROBE_FILTERs are generated smallest-first
	for (int level = 1; level <= max_level; level++) {
		std::sort(nodes_by_level[level].begin(), nodes_by_level[level].end(), [](const TreeNode *a, const TreeNode *b) {
			return a->table_op->estimated_cardinality < b->table_op->estimated_cardinality;
		});
	}

	// step 3: forward pass - bottom-up (leaves to root)
	// process levels from highest (leaves) down to 1
	for (int level = max_level; level >= 1; level--) {
		for (TreeNode *child_node : nodes_by_level[level]) {
			if (!child_node) {
				D_PRINTF("ERROR: Null child_node at level %d", level);
				continue;
			}

			TreeNode *parent_node = child_node->parent;
			if (!parent_node) {
				D_PRINTF("ERROR: Null parent_node for table %llu at level %d",
				         (unsigned long long)child_node->table_idx, level);
				continue;
			}

			JoinEdge *edge = child_node->edge_to_parent;
			if (!edge) {
				D_PRINTF("ERROR: Null edge_to_parent for table %llu", (unsigned long long)child_node->table_idx);
				continue;
			}

			// determine which columns belong to child and which to parent
			vector<ColumnBinding> child_columns, parent_columns;

			if (edge->table_a == child_node->table_idx) {
				child_columns = edge->join_columns_a;
				parent_columns = edge->join_columns_b;
			} else {
				child_columns = edge->join_columns_b;
				parent_columns = edge->join_columns_a;
			}

			// CREATE_FILTER on child
			FilterOperation create_op;
			create_op.build_table_idx = child_node->table_idx;
			create_op.probe_table_idx = parent_node->table_idx;
			create_op.build_columns = child_columns;
			create_op.probe_columns = parent_columns;
			create_op.is_create = true;
			create_op.is_forward_pass = true;
			create_op.sequence_number = sequence++;
			forward_filter_ops[child_node->table_op].push_back(create_op);

			// PROBE_FILTER on parent
			FilterOperation use_op;
			use_op.build_table_idx = child_node->table_idx;
			use_op.probe_table_idx = parent_node->table_idx;
			use_op.build_columns = child_columns;
			use_op.probe_columns = parent_columns;
			use_op.is_create = false;
			use_op.is_forward_pass = true;
			use_op.sequence_number = sequence++;
			forward_filter_ops[parent_node->table_op].push_back(use_op);
		}
	}

	// step 4: backward pass - top-down (root to leaves)
	// process levels from 1 to max_level
	for (int level = 1; level <= max_level; level++) {
		for (TreeNode *child_node : nodes_by_level[level]) {
			if (!child_node) {
				D_PRINTF("ERROR: Null child_node at level %d", level);
				continue;
			}

			TreeNode *parent_node = child_node->parent;
			if (!parent_node) {
				D_PRINTF("ERROR: Null parent_node for table %llu at level %d",
				         (unsigned long long)child_node->table_idx, level);
				continue;
			}

			JoinEdge *edge = child_node->edge_to_parent;
			if (!edge) {
				D_PRINTF("ERROR: Null edge_to_parent for table %llu", (unsigned long long)child_node->table_idx);
				continue;
			}

			// determine which columns belong to parent and which to child
			vector<ColumnBinding> parent_columns, child_columns;

			if (edge->table_a == parent_node->table_idx) {
				parent_columns = edge->join_columns_a;
				child_columns = edge->join_columns_b;
			} else {
				parent_columns = edge->join_columns_b;
				child_columns = edge->join_columns_a;
			}

			// CREATE_FILTER on parent
			FilterOperation create_op;
			create_op.build_table_idx = parent_node->table_idx;
			create_op.probe_table_idx = child_node->table_idx;
			create_op.build_columns = parent_columns;
			create_op.probe_columns = child_columns;
			create_op.is_create = true;
			create_op.sequence_number = sequence++;
			backward_filter_ops[parent_node->table_op].push_back(create_op);

			// PROBE_FILTER on child
			FilterOperation use_op;
			use_op.build_table_idx = parent_node->table_idx;
			use_op.probe_table_idx = child_node->table_idx;
			use_op.build_columns = parent_columns;
			use_op.probe_columns = child_columns;
			use_op.is_create = false;
			use_op.sequence_number = sequence++;
			backward_filter_ops[child_node->table_op].push_back(use_op);
		}
	}

	return {std::move(forward_filter_ops), std::move(backward_filter_ops)};
}

std::pair<unordered_map<LogicalOperator *, vector<FilterOperation>>,
          unordered_map<LogicalOperator *, vector<FilterOperation>>>
RobustOptimizerContextState::GenerateStageModificationsFromDAG(vector<PhysicalDAGNode *> &all_nodes,
                                                               map<ColKey, ColKey> &uf_parent,
                                                               vector<FilterOpPair> &filter_pairs) {
	unordered_map<LogicalOperator *, vector<FilterOperation>> forward_filter_ops;
	unordered_map<LogicalOperator *, vector<FilterOperation>> backward_filter_ops;

	if (all_nodes.empty()) {
		return {std::move(forward_filter_ops), std::move(backward_filter_ops)};
	}

	// group nodes by level
	unordered_map<int, vector<PhysicalDAGNode *>> nodes_by_level;
	int max_level = 0;
	for (auto *node : all_nodes) {
		nodes_by_level[node->level].push_back(node);
		max_level = std::max(max_level, node->level);
	}

	idx_t sequence = 0;

	// Printer::Print(StringUtil::Format("[DAG-GEN] %zu nodes, max_level=%d", all_nodes.size(), max_level));
	// for (auto *node : all_nodes) {
	// 	Printer::Print(StringUtil::Format("[DAG-GEN]   table_%llu (level=%d, parents=%zu, children=%zu, card=%llu)",
	// 	                                  (unsigned long long)node->table_idx, node->level, node->parents.size(),
	// 	                                  node->children.size(),
	// 	                                  (unsigned long long)node->table_op->estimated_cardinality));
	// 	for (idx_t ei = 0; ei < node->edges_to_parents.size(); ei++) {
	// 		auto &e = node->edges_to_parents[ei];
	// 		for (idx_t ci = 0; ci < e.parent_cols.size(); ci++) {
	// 			Printer::Print(StringUtil::Format(
	// 			    "[DAG-GEN]     edge[%llu] parent=%llu col(%llu.%llu) <- child=%llu col(%llu.%llu)",
	// 			    (unsigned long long)ei, (unsigned long long)e.parent_table,
	// 			    (unsigned long long)e.parent_cols[ci].table_index,
	// 			    (unsigned long long)e.parent_cols[ci].column_index, (unsigned long long)e.child_table,
	// 			    (unsigned long long)e.child_cols[ci].table_index,
	// 			    (unsigned long long)e.child_cols[ci].column_index));
	// 		}
	// 	}
	// }

	// forward pass: bottom-up (leaves to roots), levels max_level down to 1
	for (int level = max_level; level >= 1; level--) {
		for (auto *child_node : nodes_by_level[level]) {
			for (idx_t ei = 0; ei < child_node->edges_to_parents.size(); ei++) {
				auto &edge = child_node->edges_to_parents[ei];
				auto *parent_node = child_node->parents[ei];

				// CREATE_FILTER on child (build=child, probe=parent)
				FilterOperation create_op;
				create_op.build_table_idx = child_node->table_idx;
				create_op.probe_table_idx = parent_node->table_idx;
				create_op.build_columns = edge.child_cols;
				create_op.probe_columns = edge.parent_cols;
				create_op.is_create = true;
				create_op.is_forward_pass = true;
				create_op.sequence_number = sequence++;
				forward_filter_ops[child_node->table_op].push_back(create_op);

				// PROBE_FILTER on parent
				FilterOperation use_op;
				use_op.build_table_idx = child_node->table_idx;
				use_op.probe_table_idx = parent_node->table_idx;
				use_op.build_columns = edge.child_cols;
				use_op.probe_columns = edge.parent_cols;
				use_op.is_create = false;
				use_op.is_forward_pass = true;
				use_op.sequence_number = sequence++;
				forward_filter_ops[parent_node->table_op].push_back(use_op);
				filter_pairs.push_back({create_op, use_op});
			}
		}
	}

	// backward pass: top-down (roots to leaves) with broadcast optimization.
	// for each equivalence class, build the BF once at the highest ancestor (root/bridge)
	// and broadcast it to all descendants sharing that class.

	// tracks which table created the BF for each equivalence class in the backward pass.
	// key: equiv class root (from union-find)
	// value: (build_table_op, index into backward_filter_ops[build_table_op], build_table_idx, build_columns)
	struct EquivBFSource {
		LogicalOperator *build_table_op;
		idx_t create_op_index; // index into backward_filter_ops[build_table_op]
		idx_t build_table_idx;
		vector<ColumnBinding> build_columns;
	};
	map<ColKey, EquivBFSource> equiv_class_bf_source;

	for (int level = 1; level <= max_level; level++) {
		for (auto *child_node : nodes_by_level[level]) {
			// collect parent edges with their indices, sort by parent cardinality ascending
			vector<idx_t> edge_indices;
			for (idx_t ei = 0; ei < child_node->edges_to_parents.size(); ei++) {
				edge_indices.push_back(ei);
			}
			std::sort(edge_indices.begin(), edge_indices.end(), [&](idx_t a, idx_t b) {
				return child_node->parents[a]->table_op->estimated_cardinality <
				       child_node->parents[b]->table_op->estimated_cardinality;
			});

			for (auto ei : edge_indices) {
				auto &edge = child_node->edges_to_parents[ei];
				auto *parent_node = child_node->parents[ei];

				// determine the equivalence class for this edge using the first column pair
				ColKey parent_col_key = {edge.parent_cols[0].table_index, edge.parent_cols[0].column_index};
				ColKey equiv_root = UFFind(uf_parent, parent_col_key);

				auto it = equiv_class_bf_source.find(equiv_root);
				if (it != equiv_class_bf_source.end()) {
					// an ancestor already created a BF for this equivalence class — broadcast (USE only)
					auto &source = it->second;

					// add child's probe columns (and matching build columns) to the existing
					// CREATE_FILTER so linking can find it and the merge logic preserves them
					auto &create_op = backward_filter_ops[source.build_table_op][source.create_op_index];
					for (idx_t ci = 0; ci < edge.child_cols.size(); ci++) {
						create_op.probe_columns.push_back(edge.child_cols[ci]);
						create_op.build_columns.push_back(
						    source.build_columns[ci < source.build_columns.size() ? ci
						                                                          : source.build_columns.size() - 1]);
					}

					FilterOperation use_op;
					use_op.build_table_idx = source.build_table_idx;
					use_op.probe_table_idx = child_node->table_idx;
					use_op.build_columns = source.build_columns;
					use_op.probe_columns = edge.child_cols;
					use_op.is_create = false;
					use_op.is_forward_pass = false;
					use_op.sequence_number = sequence++;
					backward_filter_ops[child_node->table_op].push_back(use_op);
					filter_pairs.push_back({create_op, use_op});
				} else {
					// new equivalence class at this edge — create BF on parent, use on child
					FilterOperation create_op;
					create_op.build_table_idx = parent_node->table_idx;
					create_op.probe_table_idx = child_node->table_idx;
					create_op.build_columns = edge.parent_cols;
					create_op.probe_columns = edge.child_cols;
					create_op.is_create = true;
					create_op.is_forward_pass = false;
					create_op.sequence_number = sequence++;

					idx_t create_idx = backward_filter_ops[parent_node->table_op].size();
					backward_filter_ops[parent_node->table_op].push_back(create_op);

					FilterOperation use_op;
					use_op.build_table_idx = parent_node->table_idx;
					use_op.probe_table_idx = child_node->table_idx;
					use_op.build_columns = edge.parent_cols;
					use_op.probe_columns = edge.child_cols;
					use_op.is_create = false;
					use_op.is_forward_pass = false;
					use_op.sequence_number = sequence++;
					backward_filter_ops[child_node->table_op].push_back(use_op);
					filter_pairs.push_back({create_op, use_op});

					// record this as the source for this equivalence class
					equiv_class_bf_source[equiv_root] = {parent_node->table_op, create_idx, parent_node->table_idx,
					                                     edge.parent_cols};
				}
			}
		}
	}

	return {std::move(forward_filter_ops), std::move(backward_filter_ops)};
}

unique_ptr<LogicalOperator>
RobustOptimizerContextState::BuildStackedBFOperators(unique_ptr<LogicalOperator> base_plan,
                                                     const vector<FilterOperation> &filter_ops, bool reverse_order) {
	if (filter_ops.empty()) {
		return base_plan;
	}

	// preserve order and only merge consecutive CREATEs for the same table
	vector<FilterOperation> merged_ops;

	for (size_t i = 0; i < filter_ops.size(); i++) {
		const auto &filter_op = filter_ops[i];

		if (filter_op.is_create) {
			// Check if we can merge with subsequent consecutive CREATEs for same table
			vector<FilterOperation> consecutive_creates;
			consecutive_creates.push_back(filter_op);

			// Look ahead for consecutive CREATEs on the same table
			size_t j = i + 1;
			while (j < filter_ops.size() && filter_ops[j].is_create &&
			       filter_ops[j].build_table_idx == filter_op.build_table_idx) {
				consecutive_creates.push_back(filter_ops[j]);
				j++;
			}

			if (consecutive_creates.size() == 1) {
				// single CREATE, no merging needed
				merged_ops.push_back(filter_op);
			} else {
				// multiple consecutive CREATEs for same table - merge them
				FilterOperation merged_op = consecutive_creates[0];
				merged_op.build_columns.clear();
				merged_op.probe_columns.clear();

				// collect all build columns
				for (const auto &op : consecutive_creates) {
					// for (const auto &col : op.build_columns) {
					for (idx_t x = 0; x < op.build_columns.size(); x++) {
						// __assert(op.build_columns.size() == op.probe_columns.size(),"Merging consecutive
						// CREATE_FILTERs: Build columns and probe columns size different");
						merged_op.build_columns.push_back(op.build_columns[x]);
						merged_op.probe_columns.push_back(op.probe_columns[x]);
					}
				}
				merged_ops.push_back(merged_op);
			}

			// skip the operations we just merged
			i = j - 1;
		} else {
			// USE operation - add as is
			merged_ops.push_back(filter_op);
		}
	}

	// build operators from merged list
	unique_ptr<LogicalOperator> current = std::move(base_plan);

	// helper to set estimated_cardinality from the underlying scan table
	auto set_cardinality = [&](LogicalOperator *op, const FilterOperation &filter_op) {
		idx_t table_idx = filter_op.is_create ? filter_op.build_table_idx : filter_op.probe_table_idx;
		auto it = table_mgr.table_lookup.find(table_idx);
		if (it != table_mgr.table_lookup.end()) {
			op->estimated_cardinality = it->second.estimated_cardinality;
		}
	};

	if (reverse_order) {
		for (auto it = merged_ops.rbegin(); it != merged_ops.rend(); ++it) {
			const auto &filter_op = *it;
			unique_ptr<LogicalOperator> new_op;

			if (filter_op.is_create) {
				auto create = make_uniq<LogicalCreateFilter>(filter_op);
				create->is_forward_pass = filter_op.is_forward_pass;
				new_op = std::move(create);
			} else {
				new_op = make_uniq<LogicalProbeFilter>(filter_op);
			}

			set_cardinality(new_op.get(), filter_op);
			new_op->AddChild(std::move(current));
			current = std::move(new_op);
		}
	} else {
		for (const auto &filter_op : merged_ops) {
			unique_ptr<LogicalOperator> new_op;

			if (filter_op.is_create) {
				auto create = make_uniq<LogicalCreateFilter>(filter_op);
				create->is_forward_pass = filter_op.is_forward_pass;
				new_op = std::move(create);
			} else {
				new_op = make_uniq<LogicalProbeFilter>(filter_op);
			}

			set_cardinality(new_op.get(), filter_op);
			new_op->AddChild(std::move(current));
			current = std::move(new_op);
		}
	}
	return current;
}

unique_ptr<LogicalOperator> RobustOptimizerContextState::ApplyStageModifications(
    unique_ptr<LogicalOperator> plan,
    const unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
    const unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops) {
	// first apply modifications to children recursively
	for (auto &child : plan->children) {
		child = ApplyStageModifications(std::move(child), forward_filter_ops, backward_filter_ops);
	}

	LogicalOperator *original_op = plan.get();

	// add the forward pass bf operators above the base table operator
	auto forward_it = forward_filter_ops.find(original_op);
	if (forward_it != forward_filter_ops.end()) {
		plan = BuildStackedBFOperators(std::move(plan), forward_it->second, false);
	}

	// add the backward pass bf operators above the forward pass bf operators
	auto backward_it = backward_filter_ops.find(original_op);
	if (backward_it != backward_filter_ops.end()) {
		// for (size_t i = 0; i < backward_it->second.size(); i++) {
		// 	const auto &op = backward_it->second[i];
		// }
		plan = BuildStackedBFOperators(std::move(plan), backward_it->second, false);
	}

	return plan;
}

void RobustOptimizerContextState::LinkProbeFilterToCreateFilter(LogicalOperator *plan) {
	if (!plan) {
		return;
	}

	// helper struct to uniquely identify a CREATE_FILTER
	struct CreateFilterKey {
		idx_t build_table_idx;
		vector<ColumnBinding> build_columns;

		bool operator==(const CreateFilterKey &other) const {
			if (build_table_idx != other.build_table_idx) {
				return false;
			}
			if (build_columns.size() != other.build_columns.size()) {
				return false;
			}
			for (size_t i = 0; i < build_columns.size(); i++) {
				if (build_columns[i].table_index != other.build_columns[i].table_index ||
				    build_columns[i].column_index != other.build_columns[i].column_index) {
					return false;
				}
			}
			return true;
		}
	};

	struct CreateFilterKeyHash {
		size_t operator()(const CreateFilterKey &key) const {
			size_t hash = std::hash<idx_t>()(key.build_table_idx);
			for (const auto &col : key.build_columns) {
				hash ^= (std::hash<idx_t>()(col.table_index) << 1);
				hash ^= (std::hash<idx_t>()(col.column_index) << 2);
			}
			return hash;
		}
	};

	// pass 1: collect all CREATE_FILTER operators (multiple per build table possible)
	unordered_map<idx_t, vector<LogicalCreateFilter *>> create_filter_by_table;
	vector<LogicalOperator *> queue;
	queue.push_back(plan);

	while (!queue.empty()) {
		LogicalOperator *current = queue.back();
		queue.pop_back();

		if (current->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
			auto *create_filter = dynamic_cast<LogicalCreateFilter *>(current);
			if (create_filter) {
				create_filter_by_table[create_filter->filter_operation.build_table_idx].push_back(create_filter);
			}
		}

		for (auto &child : current->children) {
			queue.push_back(child.get());
		}
	}

	// pass 2: link all PROBE_FILTER operators to their corresponding CREATE_FILTER
	queue.clear();
	queue.push_back(plan);

	while (!queue.empty()) {
		LogicalOperator *current = queue.back();
		queue.pop_back();

		if (current->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
			auto *probe_filter = dynamic_cast<LogicalProbeFilter *>(current);
			if (probe_filter) {
				idx_t build_table_idx = probe_filter->filter_operation.build_table_idx;
				idx_t probe_table_idx = probe_filter->filter_operation.probe_table_idx;

				auto it = create_filter_by_table.find(build_table_idx);
				if (it != create_filter_by_table.end()) {
					for (auto *create_filter : it->second) {
						for (const auto &pc : create_filter->filter_operation.probe_columns) {
							if (pc.table_index == probe_table_idx) {
								probe_filter->related_create_filter = create_filter;
								create_filter->related_probe_filter.push_back(probe_filter);
								break;
							}
						}
						if (probe_filter->related_create_filter) {
							break;
						}
					}
					if (!probe_filter->related_create_filter) {
						D_PRINTF("[LINK] WARNING: No CREATE_FILTER with matching probe table for PROBE_FILTER "
						         "(build=table_%llu, probe=table_%llu)",
						         (unsigned long long)build_table_idx, (unsigned long long)probe_table_idx);
					}
				} else {
					D_PRINTF(
					    "[LINK] WARNING: No CREATE_FILTER found for PROBE_FILTER (build=table_%llu, probe=table_%llu)",
					    (unsigned long long)build_table_idx, (unsigned long long)probe_table_idx);
				}
			}
		}

		for (auto &child : current->children) {
			queue.push_back(child.get());
		}
	}
}

void RobustOptimizerContextState::SetupDynamicFilterPushdown(LogicalOperator *plan) {
	if (!plan) {
		return;
	}

	// collect all forward-pass LogicalCreateFilter operators
	vector<LogicalCreateFilter *> forward_creates;
	vector<LogicalOperator *> queue;
	queue.push_back(plan);

	while (!queue.empty()) {
		LogicalOperator *current = queue.back();
		queue.pop_back();

		if (current->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
			auto *create_filter = dynamic_cast<LogicalCreateFilter *>(current);
			if (create_filter) {
				if (create_filter->is_forward_pass) {
					forward_creates.push_back(create_filter);
				}
			}
		}

		for (auto &child : current->children) {
			queue.push_back(child.get());
		}
	}

	D_PRINTF("[PUSHDOWN-SETUP] found %zu forward CREATE_FILTERs", forward_creates.size());

	// for each forward-pass CREATE_FILTER, set up pushdown targets
	for (auto *create_filter : forward_creates) {
		D_PRINTF("[PUSHDOWN-SETUP] CREATE_FILTER build=table_%llu, related_probe_filter=%zu",
		         (unsigned long long)create_filter->filter_operation.build_table_idx,
		         create_filter->related_probe_filter.size());
		for (auto *probe_filter : create_filter->related_probe_filter) {
			if (!probe_filter->filter_operation.is_forward_pass) {
				D_PRINTF("[PUSHDOWN-SETUP]   skipping PROBE_FILTER probe=table_%llu (not forward)",
				         (unsigned long long)probe_filter->filter_operation.probe_table_idx);
				continue;
			}

			idx_t probe_table_idx = probe_filter->filter_operation.probe_table_idx;
			auto it = table_mgr.table_lookup.find(probe_table_idx);
			if (it == table_mgr.table_lookup.end()) {
				D_PRINTF("[PUSHDOWN-SETUP]   probe table_%llu not in table_lookup",
				         (unsigned long long)probe_table_idx);
				continue;
			}

			LogicalGet *get = TableManager::FindLogicalGet(it->second.table_op);
			if (!get) {
				continue;
			}

			// create or reuse DynamicTableFilterSet on the LogicalGet
			if (!get->dynamic_filters) {
				get->dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
			}

			// resolve each probe column to a scan column index
			auto &col_ids = get->GetColumnIds();
			for (size_t i = 0; i < probe_filter->filter_operation.probe_columns.size(); i++) {
				const auto &probe_col = probe_filter->filter_operation.probe_columns[i];

				idx_t scan_col_idx = probe_col.column_index;
				if (scan_col_idx >= col_ids.size()) {
					D_PRINTF("[PUSHDOWN] probe column (%llu.%llu) out of bounds for scan column_ids (size=%zu)",
					         (unsigned long long)probe_col.table_index, (unsigned long long)probe_col.column_index,
					         col_ids.size());
					continue;
				}

				// get column type and name
				LogicalType col_type = LogicalType::BIGINT;
				string col_name = "col_" + std::to_string(probe_col.column_index);
				idx_t primary_idx = col_ids[scan_col_idx].GetPrimaryIndex();
				if (primary_idx < get->returned_types.size()) {
					col_type = get->returned_types[primary_idx];
				}
				if (primary_idx < get->names.size()) {
					col_name = get->names[primary_idx];
				}

				LogicalCreateFilter::DynamicFilterTarget target;
				target.dynamic_filters = get->dynamic_filters;
				target.scan_column_index = scan_col_idx;
				target.probe_column = probe_col;
				target.column_type = col_type;
				target.column_name = col_name;
				create_filter->pushdown_targets.push_back(std::move(target));
			}

			// mark PROBE_FILTER as passthrough since filters are pushed to scan
			probe_filter->is_passthrough = true;

			D_PRINTF("[PUSHDOWN] forward CREATE_FILTER (build=table_%llu) -> PROBE_FILTER (probe=table_%llu) pushed "
			         "%zu targets",
			         (unsigned long long)create_filter->filter_operation.build_table_idx,
			         (unsigned long long)probe_table_idx, create_filter->pushdown_targets.size());
		}
	}
}

// find the deepest CREATE_FILTER in a linear chain (following child[0])
static LogicalOperator *FindDeepestCreateFilter(LogicalOperator *node) {
	LogicalOperator *deepest = nullptr;
	while (node) {
		if (node->children.empty()) {
			break;
		}
		if (node->children.size() != 1 || node->GetColumnBindings() != node->children[0]->GetColumnBindings()) {
			break;
		}
		if (node->type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
		    dynamic_cast<LogicalCreateFilter *>(node)) {
			deepest = node;
		}
		node = node->children[0].get();
	}
	return deepest;
}

void RobustOptimizerContextState::LiftCreateFilterAboveMarkJoin(unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}
	for (auto &child : plan->children) {
		LiftCreateFilterAboveMarkJoin(child);
	}

	// match: MARK JOIN with BF operators in probe chain
	if (plan->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &join = plan->Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::MARK) {
		return;
	}

	auto &probe_child = plan->children[0];
	auto *deepest = FindDeepestCreateFilter(probe_child.get());
	if (!deepest) {
		return;
	}

	// detach block from probe chain, place above MARK_JOIN
	auto below_deepest = std::move(deepest->children[0]);
	deepest->children.clear();
	auto block = std::move(probe_child);
	probe_child = std::move(below_deepest);

	deepest->AddChild(std::move(plan));
	plan = std::move(block);
}

void RobustOptimizerContextState::LiftCreateFilterAboveFilter(unique_ptr<LogicalOperator> &plan) {
	if (!plan) {
		return;
	}
	for (auto &child : plan->children) {
		LiftCreateFilterAboveFilter(child);
	}

	if (plan->type != LogicalOperatorType::LOGICAL_FILTER) {
		return;
	}

	auto *deepest = FindDeepestCreateFilter(plan->children[0].get());
	if (!deepest) {
		return;
	}

	// same block-detach logic as LiftCreateFilterAboveMarkJoin
	auto below_deepest = std::move(deepest->children[0]);
	deepest->children.clear();
	auto block = std::move(plan->children[0]);
	plan->children[0] = std::move(below_deepest);

	deepest->AddChild(std::move(plan));
	plan = std::move(block);
}

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

unique_ptr<BaseStatistics> RobustOptimizerContextState::GetColumnStatistics(const ColumnBinding &binding) {
	auto &table_info = table_mgr.table_lookup[binding.table_index];
	auto *get = TableManager::FindLogicalGet(table_info.table_op);
	if (!get) {
		D_PRINTF("No matching base table found");
		return nullptr;
	}
	auto &column_id = get->GetColumnIds()[binding.column_index];

	if (!get->function.statistics && !get->function.statistics_extended) {
		return nullptr;
	}
	unique_ptr<BaseStatistics> column_statistics;
	if (get->function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get->bind_data.get(), column_id);
		column_statistics = get->function.statistics_extended(context, input);
	} else {
		D_ASSERT(get->function.statistics);
		column_statistics = get->function.statistics(context, get->bind_data.get(), column_id.GetPrimaryIndex());
	}
	return column_statistics;
}

idx_t RobustOptimizerContextState::GetBaseTableRowCount(const ColumnBinding &binding) {
	auto resolved_binding = ResolveColumnBinding(binding);
	auto table_it = table_mgr.table_lookup.find(resolved_binding.table_index);
	if (table_it == table_mgr.table_lookup.end()) {
		return DConstants::INVALID_INDEX;
	}
	auto *get = TableManager::FindLogicalGet(table_it->second.table_op);
	if (!get || !get->function.cardinality) {
		return DConstants::INVALID_INDEX;
	}
	auto table_stats = get->function.cardinality(context, get->bind_data.get());
	if (!table_stats || !table_stats->has_estimated_cardinality) {
		return DConstants::INVALID_INDEX;
	}
	return table_stats->estimated_cardinality;
}

unique_ptr<HyperLogLog> RobustOptimizerContextState::GetColumnHLL(const ColumnBinding &binding) {
	auto resolved_binding = ResolveColumnBinding(binding);
	auto table_it = table_mgr.table_lookup.find(resolved_binding.table_index);
	if (table_it == table_mgr.table_lookup.end()) {
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
bool RobustOptimizerContextState::HLLDominates(const ColumnBinding &build_binding, const ColumnBinding &probe_binding) {
	auto build_hll = GetColumnHLL(build_binding);
	auto probe_hll = GetColumnHLL(probe_binding);
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

void RobustOptimizerContextState::TightenWithLocalPredicate(const ColumnBinding &binding, BaseStatistics &stats) {
	if (!stats.GetType().IsIntegral() || !NumericStats::HasMinMax(stats)) {
		return;
	}
	auto resolved = ResolveColumnBinding(binding);
	auto *get = TableManager::FindLogicalGet(table_mgr.table_lookup[resolved.table_index].table_op);
	if (!get) {
		D_PRINTF("No matching base table found");
		return;
	}

	auto col = get->GetColumnIds()[resolved.column_index].GetPrimaryIndex();
	auto it = get->table_filters.filters.find(col);
	if (it != get->table_filters.filters.end()) {
		TightenProbeRange(stats, *it->second);
	}
}

bool RobustOptimizerContextState::HasFilteringLocalPredicate(const FilterOpPair &pair) {
	const auto &pair_op = pair.probe_op;
	const auto &build_columns = pair_op.build_columns;
	const auto &probe_columns = pair_op.probe_columns;
	auto build_table_idx = pair_op.build_table_idx;

	if (table_with_filters.count(build_table_idx)) {
		return true;
	}

	auto &table_info = table_mgr.table_lookup[build_table_idx];
	auto *get = TableManager::FindLogicalGet(table_info.table_op);
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
		auto build_stats = GetColumnStatistics(filter_binding);
		if (!build_stats) {
			return true;
		}
		auto build_result = filter.CheckStatistics(*build_stats);
		if (build_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			continue;
		}

		// check if this local predicate is on join key
		auto resolved_filter_binding = ResolveColumnBinding(filter_binding);
		idx_t key_index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < build_columns.size(); i++) {
			if (ResolveColumnBinding(build_columns[i]) == resolved_filter_binding) {
				key_index = i;
				break;
			}
		}
		if (key_index == DConstants::INVALID_INDEX) {
			return true;
		}
		auto probe_binding = ResolveColumnBinding(probe_columns[key_index]);
		auto probe_stats = GetColumnStatistics(probe_binding);
		if (!probe_stats) {
			return true;
		}

		TightenWithLocalPredicate(probe_binding, *probe_stats);

		auto probe_result = filter.CheckStatistics(*probe_stats);
		if (probe_result != FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return true;
		}
	}

	return false;
}

bool RobustOptimizerContextState::IsRedundant(const FilterOpPair &pair) {
	const auto &build_columns = pair.probe_op.build_columns;
	const auto &probe_columns = pair.probe_op.probe_columns;

	for (idx_t i = 0; i < build_columns.size(); ++i) {
		auto build_stats = GetColumnStatistics(build_columns[i]);
		auto probe_stats = GetColumnStatistics(probe_columns[i]);
		if (!build_stats || !probe_stats) {
			return false;
		}
		if (!build_stats->GetType().IsIntegral() || !probe_stats->GetType().IsIntegral()) {
			return false;
		}
		if (!NumericStats::HasMinMax(*build_stats) || !NumericStats::HasMinMax(*probe_stats)) {
			return false;
		}

		TightenWithLocalPredicate(probe_columns[i], *probe_stats);

		auto build_min = NumericStats::Min(*build_stats).GetValue<int64_t>();
		auto build_max = NumericStats::Max(*build_stats).GetValue<int64_t>();
		auto probe_min = NumericStats::Min(*probe_stats).GetValue<int64_t>();
		auto probe_max = NumericStats::Max(*probe_stats).GetValue<int64_t>();

		// if probe range is outside of build range, then not redundant
		if (probe_min < build_min || probe_max > build_max) {
			return false;
		}
		auto build_range = build_max - build_min + 1;
		auto build_row_count = GetBaseTableRowCount(build_columns[i]);
		bool redundant_by_row_count =
		    build_row_count != DConstants::INVALID_INDEX && build_row_count == static_cast<idx_t>(build_range);
		if (redundant_by_row_count) {
			return true;
		}
		bool redundant_by_hll = HLLDominates(build_columns[i], probe_columns[i]);
		if (redundant_by_hll) {
			return true;
		}
	}
	return false;
}

void RobustOptimizerContextState::RemoveRedundantPairs(
    vector<FilterOpPair> &filter_pairs, unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
    unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops) {
	map<idx_t, FilterTableState> table_states;

	// initialzize status of tables using local predicates
	for (const auto &entry : table_mgr.table_lookup) {
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

		if (build_state.status == IncomingFilterStatus::NONE && !HasFilteringLocalPredicate(pair) &&
		    IsRedundant(pair)) {
			redundant = true;
		}

		// peel from turnaround
		if (build_state.status == IncomingFilterStatus::ONE && build_state.source_idx == pair_op.probe_table_idx &&
		    !HasFilteringLocalPredicate(pair) && IsRedundant(pair)) {
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
unique_ptr<LogicalOperator> RobustOptimizerContextState::PreOptimize(unique_ptr<LogicalOperator> plan) {
	// step 1: extract join operators
	vector<JoinEdge> edges = ExtractOperators(*plan);

	// step 2: create transfer graph using LargestRoot algorithm
	mst_edges = LargestRoot(edges);

	return plan;
}

unique_ptr<LogicalOperator> RobustOptimizerContextState::Optimize(unique_ptr<LogicalOperator> plan) {
	// step 1: extract join operators
	vector<JoinEdge> edges = ExtractOperators(*plan);

	D_PRINTF("Edges size: %zu", edges.size());
	if (edges.size() <= 1) {
		return plan;
	}

	// display physical plan DAG if enabled (before we modify the plan)
	PrintPhysicalPlanDAG(plan.get());

	// determine heuristic
	Value heuristic_val;
	string heuristic = "join_order";
	if (context.TryGetCurrentSetting("robust_heuristic", heuristic_val)) {
		heuristic = heuristic_val.GetValue<string>();
	}

	unordered_map<LogicalOperator *, vector<FilterOperation>> forward_filter_ops, backward_filter_ops;

	if (heuristic == "join_order") {
		// use DuckDB's join order DAG
		map<ColKey, ColKey> uf_parent;
		auto all_nodes = BuildPhysicalPlanDAG(plan.get(), uf_parent);
		vector<FilterOpPair> filter_pairs;

		// flip non-largest roots to leaves (default: on)
		Value flip_val;
		bool flip_roots = true;
		if (context.TryGetCurrentSetting("robust_flip_roots", flip_val)) {
			flip_roots = flip_val.GetValue<bool>();
		}
		if (flip_roots) {
			FlipRootsToLeaves(all_nodes);
		}

		// display DAG if setting is enabled
		Value dag_val;
		if (context.TryGetCurrentSetting("robust_display_dag", dag_val) && dag_val.GetValue<bool>()) {
			PrintPhysicalDAG(all_nodes, table_mgr);
		}

		auto filter_ops = GenerateStageModificationsFromDAG(all_nodes, uf_parent, filter_pairs);
		forward_filter_ops = std::move(filter_ops.first);
		backward_filter_ops = std::move(filter_ops.second);

		RemoveRedundantPairs(filter_pairs, forward_filter_ops, backward_filter_ops);

	} else {
		// largest_root
		mst_edges = LargestRoot(edges);
		auto filter_ops = GenerateStageModifications(mst_edges);
		forward_filter_ops = std::move(filter_ops.first);
		backward_filter_ops = std::move(filter_ops.second);
	}

	// check pass mode setting
	Value pass_mode_val;
	string pass_mode = "both";
	if (context.TryGetCurrentSetting("robust_pass_mode", pass_mode_val)) {
		pass_mode = pass_mode_val.GetValue<string>();
	}
	if (pass_mode == "forward_only") {
		backward_filter_ops.clear();
	}

	// step 4: insert create_filter/probe_filter operators into the plan
	plan = ApplyStageModifications(std::move(plan), forward_filter_ops, backward_filter_ops);

	// step 4.5a: lift BF operators above MARK_JOIN so they sit between FILTER and MARK_JOIN
	LiftCreateFilterAboveMarkJoin(plan);

	// step 4.5b: lift BF operators above FILTER so bloom filters are built from filtered output
	LiftCreateFilterAboveFilter(plan);

	// step 5: link PROBE_FILTER operators to their corresponding CREATE_FILTER operators
	LinkProbeFilterToCreateFilter(plan.get());

	// step 6: set up dynamic filter pushdown for forward-pass operators
	SetupDynamicFilterPushdown(plan.get());

	// // combine all bloom filter operations for debug (preserving order)
	// vector<FilterOperation> all_filter_operations;
	// for (const auto &pair : filter_ops.first) {
	// 	all_filter_operations.insert(all_filter_operations.end(), pair.second.begin(), pair.second.end());
	// }
	// for (const auto &pair : filter_ops.second) {
	// 	all_filter_operations.insert(all_filter_operations.end(), pair.second.begin(), pair.second.end());
	// }
	//
	// // sort by sequence number to restore generation order
	// std::sort(all_filter_operations.begin(), all_filter_operations.end(),
	// 	[](const FilterOperation &a, const FilterOperation &b) {
	// 		return a.sequence_number < b.sequence_number;
	// 	});
	//
	// // debug print with correct ordering
	// DebugPrintMST(mst_edges, all_filter_operations);
	return plan;
}

// extension hooks
// void PredicateTransferOptimizer::PreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
// 	// create optimizer state using proper DuckDB state management
// 	auto optimizer_state = input.context.registered_state->GetOrCreate<PredicateTransferOptimizer>(
// 		"robust_optimizer_state", input.context);
//
// 	plan = optimizer_state->PreOptimize(std::move(plan));
// }

void RobustOptimizerContextState::PreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto optimizer_state = input.context.registered_state->GetOrCreate<RobustOptimizerContextState>(
	    "robust_optimizer_state", input.context);

	plan = optimizer_state->PreOptimize(std::move(plan));
}

void RobustOptimizerContextState::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto profiling = GetRobustProfilingState(input.context);
	auto opt_start = std::chrono::high_resolution_clock::now();

	const auto optimizer_state = input.context.registered_state->GetOrCreate<RobustOptimizerContextState>(
	    "robust_optimizer_state", input.context);
	plan = optimizer_state->Optimize(std::move(plan));

	if (profiling) {
		auto opt_end = std::chrono::high_resolution_clock::now();
		profiling->optimizer_time_us =
		    std::chrono::duration_cast<std::chrono::microseconds>(opt_end - opt_start).count();

		// populate table names for profiling output
		for (const auto &ti : optimizer_state->table_mgr.table_ops) {
			profiling->table_names[ti.table_idx] = optimizer_state->table_mgr.GetTableName(ti.table_idx);
		}
	}

	input.context.registered_state->Remove("robust_optimizer_state");
}

} // namespace duckdb
