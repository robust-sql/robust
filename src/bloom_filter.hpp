//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/predicate_transfer/bloom_filter/bloom_filter.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include <cstdint>
#include <atomic>

namespace duckdb {

class ColumnDataCollection;

// wrapper around DuckDB's native BloomFilter with DataChunk-level operations
class PTBloomFilter {
public:
	PTBloomFilter() = default;
	void Initialize(ClientContext &context_p, uint32_t est_num_rows);

	ClientContext *context = nullptr;
	BufferManager *buffer_manager = nullptr;

	bool finalized_ = false;

public:
	idx_t LookupSel(DataChunk &chunk, SelectionVector &sel, const vector<idx_t> &bound_cols_applied,
	                uint8_t *bit_vector_buf) const;
	void Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built);

	// reallocate the native BF for `actual_rows` and re-hash all rows from `data` on `cols`
	void ReinitializeAndRehash(ClientContext &context_p, idx_t actual_rows, ColumnDataCollection &data,
	                           const vector<idx_t> &cols);

	idx_t SizedForRows() const {
		return sized_for_rows_;
	}

	bool IsEmpty() const {
		return !has_data_;
	}

	shared_ptr<const BloomFilter> GetNativeFilter() const {
		return bf_;
	}

	void Seal(ClientContext &context) {
		if (!IsEmpty()) {
			bf_->Seal(context);
		}
		finalized_ = true;
	}

private:
	std::atomic<bool> has_data_ {false};
	idx_t sized_for_rows_ = 0;
	shared_ptr<BloomFilter> bf_ = make_shared_ptr<BloomFilter>();
};

} // namespace duckdb
