// include/strata/core/exl3_flags.hpp - EXL3 environment-switch lookup and defaults.
//
// Lookup order per switch: the STRATA_EXL3_<NAME> value when set and non-empty,
// then the legacy shared name (when given) on the same terms, then null.  An
// empty value counts as unset; "0" is an explicit off, any other value on.
//
// Shared switches are initialized only for an EXL3 process, before their
// first use; GGUF processes retain their existing defaults.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The switch's value, or null when unset: `exl3_name` first, then `shared_name`.
inline const char* exl3_env(const char* exl3_name, const char* shared_name = nullptr) {
    if (exl3_name != nullptr) {
        if (const char* v = std::getenv(exl3_name); v != nullptr && v[0] != '\0') return v;
    }
    if (shared_name != nullptr) {
        if (const char* v = std::getenv(shared_name); v != nullptr && v[0] != '\0') return v;
    }
    return nullptr;
}

/// A boolean switch: "0" off, any other non-empty value on, `default_on` when unset.
inline bool exl3_flag(const char* exl3_name, const char* shared_name, bool default_on) {
    const char* v = exl3_env(exl3_name, shared_name);
    return v == nullptr ? default_on : v[0] != '0';
}

/// Resolve EXL3-scoped aliases before shared switch readers cache their values.
/// Call only for an EXL3 process; scoped value beats legacy value, then default 1.
inline std::vector<std::string> apply_exl3_defaults() {
    static const std::pair<const char*, const char*> table[] = {
        {"STRATA_EXL3_QFUSE", "STRATA_QFUSE"},
        {"STRATA_EXL3_MMVQ_BUNCH", "STRATA_MMVQ_BUNCH"},
        {"STRATA_EXL3_PF_GEMM", "STRATA_PF_GEMM"},
    };
    std::vector<std::string> set;
    for (const auto& [scoped, shared] : table) {
        const char* resolved = exl3_env(scoped, shared);
        const std::string value = resolved ? resolved : "1";
        const char* current = std::getenv(shared);
        if (current && value == current) continue;
#if defined(_WIN32)
        _putenv_s(shared, value.c_str());
#else
        setenv(shared, value.c_str(), 1);
#endif
        set.emplace_back(shared);
    }
    return set;
}

/// Qwen's tested fast path, selected by the engine, not by a shell recipe.
/// Only the measured gfx1100 family is selected automatically. Other backends
/// retain their previous dispatch. No memory, CPU, topology or repack policy here.
inline std::vector<std::string> apply_exl3_fast_defaults(bool gfx1100) {
    std::vector<std::string> changed;
    if (!gfx1100) return changed;
    const std::pair<const char*, const char*> defaults[] = {
        {"STRATA_EXL3_RDNA", "1"}, {"STRATA_EXL3_FUSE", "3"},
        {"STRATA_EXL3_WMMA", "1"}, {"STRATA_WMMA_GEMM", "1"},
        {"STRATA_HIP_WMMA", "1"}, {"STRATA_EXL3_RDNA_NATIVE", "1"},
        {"STRATA_EXL3_NATIVE_GROUPED", "1"}, {"STRATA_EXL3_WMMA_STG_PAD", "1"},
        {"STRATA_EXL3_PRIMARY_MULTIROW", "2"}, {"STRATA_EXL3_PEER_MULTIROW", "4"},
        {"STRATA_EXL3_EXPERT_BATCH", "1"}, {"STRATA_EXL3_PEER_BATCH", "1"},
        {"STRATA_EXL3_RESIDENT_DIRECT_DMA", "1"},
    };
    for (const auto& [key, value] : defaults) {
        if (exl3_env(key)) continue; // explicit 0 is a real override
#if defined(_WIN32)
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
        changed.emplace_back(key);
    }
    return changed;
}

}  // namespace strata::core
