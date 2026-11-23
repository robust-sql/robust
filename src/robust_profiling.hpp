#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include <chrono>
#include <ctime>
#include <atomic>
#include <mutex>
#include <vector>
#include <algorithm>
#include <map>
#ifdef _WIN32
#include "duckdb/common/windows.hpp"
#endif

namespace duckdb {

struct CreateFilterStats {
	idx_t sequence_number = 0;
	idx_t build_table_idx = 0;
	vector<idx_t> probe_table_indices;
	bool is_forward_pass = false;
	std::atomic<idx_t> rows_materialized {0};
	std::atomic<int64_t> sink_time_us {0};
	std::atomic<int64_t> finalize_time_us {0};
	std::atomic<int64_t> source_time_us {0};
};

struct ProbeFilterStats {
	idx_t sequence_number = 0;
	idx_t build_table_idx = 0;
	idx_t probe_table_idx = 0;
	bool is_forward_pass = false;
	std::atomic<idx_t> rows_in {0};
	std::atomic<idx_t> rows_out {0};
	std::atomic<int64_t> probe_time_us {0};
};

// RAII timer that adds elapsed microseconds to an atomic counter
struct ScopedTimer {
	std::atomic<int64_t> &target;
	std::chrono::high_resolution_clock::time_point start;

	explicit ScopedTimer(std::atomic<int64_t> &target) : target(target) {
		start = std::chrono::high_resolution_clock::now();
	}
	~ScopedTimer() {
		auto end = std::chrono::high_resolution_clock::now();
		target.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(),
		                 std::memory_order_relaxed);
	}
};

// RAII timer that adds elapsed *thread CPU* microseconds to an atomic counter.
// uses CLOCK_THREAD_CPUTIME_ID (posix) / GetThreadTimes (windows) so the measurement reflects
// work actually executed and is not inflated by thread descheduling/contention (which
// wall-clock would capture). note: windows granularity is coarse (~15ms scheduler tick).
struct ScopedCpuTimer {
	std::atomic<int64_t> &target;
	int64_t start_us;

	static int64_t ThreadCpuUs() {
#ifdef _WIN32
		FILETIME creation_time, exit_time, kernel_time, user_time;
		GetThreadTimes(GetCurrentThread(), &creation_time, &exit_time, &kernel_time, &user_time);
		auto to_us = [](const FILETIME &ft) {
			return ((static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) / 10;
		};
		return to_us(kernel_time) + to_us(user_time);
#else
		timespec ts;
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
		return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
#endif
	}

	explicit ScopedCpuTimer(std::atomic<int64_t> &target) : target(target), start_us(ThreadCpuUs()) {
	}
	~ScopedCpuTimer() {
		target.fetch_add(ThreadCpuUs() - start_us, std::memory_order_relaxed);
	}
};

class RobustProfilingState : public ClientContextState {
public:
	explicit RobustProfilingState(bool enabled) : enabled(enabled) {
	}

	bool enabled;
	int64_t optimizer_time_us = 0;

	// table index -> resolved table name (populated at optimizer time)
	std::map<idx_t, string> table_names;

	mutex stats_lock;
	vector<shared_ptr<CreateFilterStats>> create_filter_stats;
	vector<shared_ptr<ProbeFilterStats>> probe_filter_stats;

	string GetName(idx_t table_idx) const {
		auto it = table_names.find(table_idx);
		if (it != table_names.end()) {
			return it->second;
		}
		return "table_" + std::to_string(table_idx);
	}

	shared_ptr<CreateFilterStats> RegisterCreateFilter(idx_t build_table_idx,
	                                                   const vector<ColumnBinding> &probe_columns,
	                                                   idx_t sequence_number, bool is_forward_pass) {
		lock_guard<mutex> lock(stats_lock);
		auto stats = make_shared_ptr<CreateFilterStats>();
		stats->sequence_number = sequence_number;
		stats->build_table_idx = build_table_idx;
		stats->is_forward_pass = is_forward_pass;
		// extract unique probe table indices from probe columns
		for (const auto &col : probe_columns) {
			if (stats->probe_table_indices.empty() || stats->probe_table_indices.back() != col.table_index.index) {
				// check if already present
				bool found = false;
				for (auto idx : stats->probe_table_indices) {
					if (idx == col.table_index.index) {
						found = true;
						break;
					}
				}
				if (!found) {
					stats->probe_table_indices.push_back(col.table_index.index);
				}
			}
		}
		create_filter_stats.push_back(stats);
		return stats;
	}

	shared_ptr<ProbeFilterStats> RegisterProbeFilter(idx_t build_table_idx, idx_t probe_table_idx,
	                                                 idx_t sequence_number, bool is_forward_pass) {
		lock_guard<mutex> lock(stats_lock);
		auto stats = make_shared_ptr<ProbeFilterStats>();
		stats->sequence_number = sequence_number;
		stats->build_table_idx = build_table_idx;
		stats->probe_table_idx = probe_table_idx;
		stats->is_forward_pass = is_forward_pass;
		probe_filter_stats.push_back(stats);
		return stats;
	}

	void QueryEnd(ClientContext &context) override {
		if (!enabled) {
			return;
		}
		PrintSummary();
		context.registered_state->Remove("robust_profiling");
	}

	void PrintSummary() {
		Printer::Print("\n=== Robust PROFILING ===");
		Printer::PrintF("Optimizer: %lld us", (int64_t)optimizer_time_us);

		// build a combined list sorted by sequence_number
		struct StatsEntry {
			idx_t seq;
			bool is_create;
			size_t idx;
		};
		vector<StatsEntry> entries;
		for (size_t i = 0; i < create_filter_stats.size(); i++) {
			entries.push_back({create_filter_stats[i]->sequence_number, true, i});
		}
		for (size_t i = 0; i < probe_filter_stats.size(); i++) {
			entries.push_back({probe_filter_stats[i]->sequence_number, false, i});
		}
		std::sort(entries.begin(), entries.end(),
		          [](const StatsEntry &a, const StatsEntry &b) { return a.seq < b.seq; });

		// per-pass accumulators
		int64_t fwd_rows_in = 0, fwd_rows_out = 0, fwd_probe_us = 0;
		int64_t bwd_rows_in = 0, bwd_rows_out = 0, bwd_probe_us = 0;
		int64_t total_sink_us = 0, total_source_us = 0, total_finalize_us = 0;

		Printer::Print("");
		for (auto &e : entries) {
			if (e.is_create) {
				auto &s = create_filter_stats[e.idx];
				string pass = s->is_forward_pass ? "FWD" : "BWD";
				string probe_names;
				for (size_t pi = 0; pi < s->probe_table_indices.size(); pi++) {
					if (pi > 0) {
						probe_names += ",";
					}
					probe_names += GetName(s->probe_table_indices[pi]);
				}
				if (probe_names.empty()) {
					probe_names = "?";
				}
				Printer::PrintF(
				    "CREATE_FILTER [%s]: [build=%s -> probe=%s] %llu rows, sink=%lldus, finalize=%lldus, source=%lldus",
				    pass.c_str(), GetName(s->build_table_idx).c_str(), probe_names.c_str(),
				    (uint64_t)s->rows_materialized.load(), (int64_t)s->sink_time_us.load(),
				    (int64_t)s->finalize_time_us.load(), (int64_t)s->source_time_us.load());
				total_sink_us += s->sink_time_us.load();
				total_source_us += s->source_time_us.load();
				total_finalize_us += s->finalize_time_us.load();
			} else {
				auto &s = probe_filter_stats[e.idx];
				string pass = s->is_forward_pass ? "FWD" : "BWD";
				idx_t ri = s->rows_in.load();
				idx_t ro = s->rows_out.load();
				double sel = ri > 0 ? 100.0 * static_cast<double>(ro) / static_cast<double>(ri) : 0.0;
				Printer::PrintF(
				    "PROBE_FILTER    [%s]: [build=%s, probe=%s] in=%llu, out=%llu, sel=%.1f%%, probe=%lldus",
				    pass.c_str(), GetName(s->build_table_idx).c_str(), GetName(s->probe_table_idx).c_str(),
				    (uint64_t)ri, (uint64_t)ro, sel, (int64_t)s->probe_time_us.load());
				if (s->is_forward_pass) {
					fwd_rows_in += static_cast<int64_t>(ri);
					fwd_rows_out += static_cast<int64_t>(ro);
					fwd_probe_us += s->probe_time_us.load();
				} else {
					bwd_rows_in += static_cast<int64_t>(ri);
					bwd_rows_out += static_cast<int64_t>(ro);
					bwd_probe_us += s->probe_time_us.load();
				}
			}
		}

		Printer::Print("\nTotals:");
		Printer::PrintF("  sink: %lld us", (int64_t)total_sink_us);
		Printer::PrintF("  source: %lld us", (int64_t)total_source_us);
		Printer::PrintF("  finalize (BF build): %lld us", (int64_t)total_finalize_us);

		auto print_pass_stats = [](const char *label, int64_t rows_in, int64_t rows_out, int64_t probe_us) {
			if (rows_in > 0) {
				double filtered_pct = 100.0 * (1.0 - static_cast<double>(rows_out) / static_cast<double>(rows_in));
				Printer::PrintF("  %s probe: %lld us, filtered: %lld / %lld rows (%.1f%% removed)", label,
				                (int64_t)probe_us, (int64_t)(rows_in - rows_out), (int64_t)rows_in, filtered_pct);
			}
		};

		print_pass_stats("forward", fwd_rows_in, fwd_rows_out, fwd_probe_us);
		print_pass_stats("backward", bwd_rows_in, bwd_rows_out, bwd_probe_us);

		int64_t total_rows_in = fwd_rows_in + bwd_rows_in;
		int64_t total_rows_out = fwd_rows_out + bwd_rows_out;
		if (total_rows_in > 0) {
			double filtered_pct =
			    100.0 * (1.0 - static_cast<double>(total_rows_out) / static_cast<double>(total_rows_in));
			Printer::PrintF("  total probe: %lld us, filtered: %lld / %lld rows (%.1f%% removed)",
			                (int64_t)(fwd_probe_us + bwd_probe_us), (int64_t)(total_rows_in - total_rows_out),
			                (int64_t)total_rows_in, filtered_pct);
		}
		Printer::Print("=== END Robust PROFILING ===\n");
	}
};

inline shared_ptr<RobustProfilingState> GetRobustProfilingState(ClientContext &context) {
	Value val;
	auto result = context.TryGetCurrentSetting("robust_profiling", val);
	if (result && val.GetValue<bool>()) {
		return context.registered_state->GetOrCreate<RobustProfilingState>("robust_profiling", true);
	}
	return nullptr;
}

} // namespace duckdb
