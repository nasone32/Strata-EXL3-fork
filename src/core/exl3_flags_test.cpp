// src/core/exl3_flags_test.cpp - the EXL3 switch lookup and defaults (no GPU needed).
#include "strata/core/exl3_flags.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    const char* exl3 = "STRATA_EXL3_TEST_SWITCH";
    const char* shared = "STRATA_TEST_SHARED_SWITCH";

    // exl3_env: the EXL3 name wins, then the shared name, then null.
    unset(exl3); unset(shared);
    CHECK(exl3_env(exl3, shared) == nullptr);
    CHECK(exl3_env(exl3) == nullptr);
    CHECK(exl3_env(nullptr, shared) == nullptr);
    put(shared, "1");
    CHECK(exl3_env(exl3, shared) != nullptr && std::string(exl3_env(exl3, shared)) == "1");
    put(exl3, "0");   // an explicit 0 is a VALUE, not an unset
    CHECK(exl3_env(exl3, shared) != nullptr && std::string(exl3_env(exl3, shared)) == "0");
    put(exl3, "2");
    CHECK(std::string(exl3_env(exl3, shared)) == "2");
    // an EMPTY value counts as unset: the alias is read, then the default.
    put(exl3, "");
    CHECK(std::string(exl3_env(exl3, shared)) == "1");
    put(shared, "");
    CHECK(exl3_env(exl3, shared) == nullptr);

    // exl3_flag: "0" off, any other non-empty value on, `default_on` when unset.
    unset(exl3); unset(shared);
    CHECK(exl3_flag(exl3, shared, true));
    CHECK(!exl3_flag(exl3, shared, false));
    put(exl3, "0");
    CHECK(!exl3_flag(exl3, shared, true));    // the explicit 0 beats the default-on
    CHECK(!exl3_flag(exl3, shared, false));
    put(exl3, "1");
    CHECK(exl3_flag(exl3, shared, false));
    put(exl3, "yes");
    CHECK(exl3_flag(exl3, shared, false));
    unset(exl3);
    put(shared, "0");                          // the alias carries its own explicit 0
    CHECK(!exl3_flag(exl3, shared, true));
    put(shared, "1");
    CHECK(exl3_flag(exl3, shared, false));
    unset(shared);

    // apply_exl3_defaults: sets only the unset GGUF-shared names, keeps explicit values.
    unset("STRATA_QFUSE"); unset("STRATA_MMVQ_BUNCH"); unset("STRATA_PF_GEMM");
    unset("STRATA_EXL3_QFUSE"); unset("STRATA_EXL3_MMVQ_BUNCH"); unset("STRATA_EXL3_PF_GEMM");
    {
        const auto set = apply_exl3_defaults();
        CHECK(set.size() == 3);
        CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
        CHECK(std::getenv("STRATA_MMVQ_BUNCH") != nullptr &&
              std::string(std::getenv("STRATA_MMVQ_BUNCH")) == "1");
        CHECK(apply_exl3_defaults().empty());   // already set: nothing more to do
    }
    put("STRATA_QFUSE", "0");                   // a user's explicit off is never overridden
    unset("STRATA_MMVQ_BUNCH");
    {
        const auto set = apply_exl3_defaults();
        CHECK(set.size() == 1);
        CHECK(std::string(std::getenv("STRATA_QFUSE")) == "0");
        CHECK(std::string(std::getenv("STRATA_MMVQ_BUNCH")) == "1");
    }
    put("STRATA_MMVQ_BUNCH", "0");
    CHECK(apply_exl3_defaults().empty());
    CHECK(std::string(std::getenv("STRATA_MMVQ_BUNCH")) == "0");
    put("STRATA_EXL3_QFUSE", "1");
    CHECK(apply_exl3_defaults().size() == 1);
    CHECK(std::string(std::getenv("STRATA_QFUSE")) == "1");
    put("STRATA_EXL3_QFUSE", "0");
    CHECK(apply_exl3_defaults().size() == 1);
    CHECK(std::string(std::getenv("STRATA_QFUSE")) == "0");
    put("STRATA_EXL3_PF_GEMM", "0");
    CHECK(apply_exl3_defaults().size() == 1);
    CHECK(std::string(std::getenv("STRATA_PF_GEMM")) == "0");
    unset("STRATA_QFUSE"); unset("STRATA_MMVQ_BUNCH"); unset("STRATA_PF_GEMM");
    unset("STRATA_EXL3_QFUSE"); unset("STRATA_EXL3_MMVQ_BUNCH"); unset("STRATA_EXL3_PF_GEMM");

    std::printf(fails ? "exl3_flags_test: %d FAILED\n" : "exl3_flags_test: OK\n", fails);
    return fails ? 1 : 0;
}