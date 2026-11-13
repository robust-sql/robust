#include "table_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "utils/debug_utils.hpp"

namespace duckdb {

// helper to get operator type name for debug
static const char *GetOpTypeName(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_GET:
		return "GET";
	case LogicalOperatorType::LOGICAL_FILTER:
		return "FILTER";
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return "PROJECTION";
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		return "COMPARISON_JOIN";
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return "DELIM_JOIN";
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return "AGGREGATE";
	case LogicalOperatorType::LOGICAL_WINDOW:
		return "WINDOW";
	case LogicalOperatorType::LOGICAL_UNION:
		return "UNION";
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		return "CHUNK_GET";
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		return "DELIM_GET";
	default:
		return "OTHER";
	}
}

void TableManager::AddTable(const TableInfo &table) {
	table_lookup[table.table_idx] = table;
	table_ops.push_back(table);
}

idx_t TableManager::GetScalarTableIndex(LogicalOperator *op) {
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		return op->GetTableIndex()[0].index;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		return GetScalarTableIndex(op->children[0].get());
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		return op->GetTableIndex()[1].index;
	}
	default:
		return std::numeric_limits<idx_t>::max();
	}
}

void TableManager::AddTableOperator(LogicalOperator *op) {
	TableInfo tbl_info;
	tbl_info.estimated_cardinality = op->estimated_cardinality;
	tbl_info.table_idx = GetScalarTableIndex(op);
	table_id table_idx = tbl_info.table_idx;
	tbl_info.table_op = op;

	if (table_idx != std::numeric_limits<idx_t>::max() && table_lookup.find(table_idx) == table_lookup.end()) {
		D_PRINTF("[NODE_REG] AddTableOperator: type=%s, table_idx=%llu, cardinality=%llu", GetOpTypeName(op->type),
		         (unsigned long long)table_idx, (unsigned long long)tbl_info.estimated_cardinality);
		table_lookup[table_idx] = tbl_info;
		table_ops.push_back(tbl_info);
	} else if (table_idx != std::numeric_limits<idx_t>::max()) {
		D_PRINTF("[NODE_REG] AddTableOperator SKIPPED (already exists): type=%s, table_idx=%llu",
		         GetOpTypeName(op->type), (unsigned long long)table_idx);
	}
}

TableInfo *TableManager::GetTableInfo(LogicalOperator *op) {
	if (!op) {
		return nullptr;
	}

	idx_t table_idx = GetScalarTableIndex(op);
	if (table_lookup.find(table_idx) == table_lookup.end()) {
		return nullptr;
	}
	return &table_lookup[table_idx];
}

LogicalGet *TableManager::FindLogicalGet(LogicalOperator *op) {
	if (!op) {
		return nullptr;
	}
	if (op->type == LogicalOperatorType::LOGICAL_GET) {
		return &op->Cast<LogicalGet>();
	}
	for (auto &child : op->children) {
		auto *result = FindLogicalGet(child.get());
		if (result) {
			return result;
		}
	}
	return nullptr;
}

string TableManager::GetTableName(idx_t table_idx) {
	auto it = table_lookup.find(table_idx);
	if (it == table_lookup.end()) {
		return "table_" + std::to_string(table_idx);
	}
	auto *get = FindLogicalGet(it->second.table_op);
	if (get) {
		auto table = get->GetTable();
		if (table) {
			return table->name.GetIdentifierName();
		}
	}
	return "table_" + std::to_string(table_idx);
}

string TableManager::GetColumnName(idx_t table_idx, idx_t column_index) {
	auto it = table_lookup.find(table_idx);
	if (it == table_lookup.end()) {
		return "col_" + std::to_string(column_index);
	}
	auto *get = FindLogicalGet(it->second.table_op);
	if (!get) {
		return "col_" + std::to_string(column_index);
	}
	auto &col_ids = get->GetColumnIds();
	if (column_index < col_ids.size()) {
		idx_t primary_idx = col_ids[column_index].GetPrimaryIndex();
		if (primary_idx < get->names.size()) {
			return get->names[primary_idx].GetIdentifierName();
		}
	}
	return "col_" + std::to_string(column_index);
}

} // namespace duckdb
