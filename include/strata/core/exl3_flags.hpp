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

}  // namespace strata::core