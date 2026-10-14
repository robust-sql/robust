#pragma once

#include "graph_manager.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class RobustOptimizerContextState;

// drop bloom filter pairs that provably cannot reject any probe-side row, so the plan does not
// pay to build and probe them. the pairs are visited in transfer order and the removed ones are
// erased from the filter op maps before the operators are inserted into the plan
void RemoveRedundantPairs(RobustOptimizerContextState &state, vector<FilterOpPair> &filter_pairs,
                          unordered_map<LogicalOperator *, vector<FilterOperation>> &forward_filter_ops,
                          unordered_map<LogicalOperator *, vector<FilterOperation>> &backward_filter_ops);

} // namespace duckdb
