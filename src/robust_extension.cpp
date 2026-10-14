#define DUCKDB_EXTENSION_MAIN

#include "robust_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator_extension.hpp"
#include "operators/logical_create_filter.hpp"
#include "operators/logical_probe_filter.hpp"
#include "optimizer/robust_optimizer.hpp"
#include "duckdb/main/config.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

class CreateFilterOperatorExtension : public OperatorExtension {
public:
	std::string GetName() override {
		return "logical_create_filter";
	}

	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &deserializer) override {
		return make_uniq<LogicalCreateFilter>();
	}
};

class ProbeFilterOperatorExtension : public OperatorExtension {
public:
	std::string GetName() override {
		return "logical_probe_filter";
	}

	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &deserializer) override {
		return make_uniq<LogicalProbeFilter>();
	}
};

static void LoadInternal(ExtensionLoader &loader) {
	// Register the SIP optimizer rule
	OptimizerExtension optimizer;
	// optimizer.pre_optimize_function = RobustOptimizerContextState::PreOptimize;
	optimizer.optimize_function = RobustOptimizerContextState::Optimize;

	DatabaseInstance &instance = loader.GetDatabaseInstance();
	OptimizerExtension::Register(instance.config, optimizer);

	// Register logical operators
	OperatorExtension::Register(instance.config, make_shared_ptr<CreateFilterOperatorExtension>());
	OperatorExtension::Register(instance.config, make_shared_ptr<ProbeFilterOperatorExtension>());

	// Register profiling setting
	auto &config = DBConfig::GetConfig(instance);
	config.AddExtensionOption("robust_profiling", "Enable Robust extension profiling output", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));
	config.AddExtensionOption("robust_display_dag", "Display Robust transfer DAG", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));
	config.AddExtensionOption("robust_display_physical_dag", "Display DAG from DuckDB join order", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));
	config.AddExtensionOption("robust_filter_type", "Filter type for scan pushdown: all, bf_only, minmax_only",
	                          LogicalType::VARCHAR, Value("all"));
	config.AddExtensionOption("robust_pass_mode", "Pass mode: both, forward_only", LogicalType::VARCHAR, Value("both"));
	config.AddExtensionOption("robust_heuristic", "Heuristic for BF transfer: join_order (default), largest_root",
	                          LogicalType::VARCHAR, Value("join_order"));
	config.AddExtensionOption("robust_flip_roots", "Flip non-largest roots to leaves in join_order DAG",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption("robust_remove_redundant_pairs",
	                          "Remove bloom filter pairs that provably cannot filter anything", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(true));
	config.AddExtensionOption("robust_dynamic_or_filter_threshold",
	                          "Max distinct build keys to push as IN-filter instead of bloom filter",
	                          LogicalType::UBIGINT, Value::UBIGINT(50));
}

void RobustExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string RobustExtension::Name() {
	return "robust";
}

std::string RobustExtension::Version() const {
#ifdef EXT_VERSION_ROBUST
	return EXT_VERSION_ROBUST;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(robust, loader) {
	duckdb::LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *robust_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
