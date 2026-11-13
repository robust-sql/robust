#include "bloom_filter.hpp"

#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

namespace {
static Vector HashColumns(DataChunk &chunk, const vector<idx_t> &cols) {
	auto count = chunk.size();
	Vector hashes(LogicalType::HASH);
	VectorOperations::Hash(chunk.data[cols[0]], hashes, count);
	for (size_t j = 1; j < cols.size(); j++) {
		VectorOperations::CombineHash(hashes, chunk.data[cols[j]], count);
	}

	if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		hashes.Flatten();
	}

	return hashes;
}
} // namespace

void PTBloomFilter::Initialize(ClientContext &context_p, uint32_t est_num_rows) {
	context = &context_p;
	buffer_manager = &BufferManager::GetBufferManager(*context);
	bf_ = make_shared_ptr<BloomFilter>();
	bf_->Initialize(context_p, static_cast<idx_t>(est_num_rows));
	sized_for_rows_ = static_cast<idx_t>(est_num_rows);
}

void PTBloomFilter::ReinitializeAndRehash(ClientContext &context_p, idx_t actual_rows, ColumnDataCollection &data,
                                          const vector<idx_t> &cols) {
	// Keep previously published filters immutable while a scan still owns a reference.
	bf_ = make_shared_ptr<BloomFilter>();
	bf_->Initialize(context_p, actual_rows);
	sized_for_rows_ = actual_rows;
	has_data_ = data.Count() > 0;

	if (data.Count() == 0) {
		return;
	}

	DataChunk chunk;
	data.InitializeScanChunk(chunk);
	ColumnDataScanState scan_state;
	data.InitializeScan(scan_state);
	while (data.Scan(scan_state, chunk)) {
		const idx_t count = chunk.size();
		if (count == 0) {
			continue;
		}
		Vector hashes = HashColumns(chunk, cols);
		bf_->InsertHashes(hashes);
	}
}

idx_t PTBloomFilter::LookupSel(DataChunk &chunk, SelectionVector &sel, const vector<idx_t> &bound_cols_applied,
                               uint8_t *bit_vector_buf) const {
	idx_t count = chunk.size();
	if (count == 0) {
		return 0;
	}
	Vector hashes = HashColumns(chunk, bound_cols_applied);
	return bf_->LookupHashes(hashes, sel, count);
}

void PTBloomFilter::Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built) {
	idx_t count = chunk.size();
	if (count == 0) {
		return;
	}
	has_data_ = true;
	Vector hashes = HashColumns(chunk, bound_cols_built);
	bf_->InsertHashes(hashes);
}

} // namespace duckdb
