from talos.diagnostics import dead_new_functions, defined_functions, relevant

FOREIGN_WARNING = (
    "warning: unused variable: `snap`\n"
    "    --> tig-algorithms/src/knapsack/superfast_knap_v1/track5.rs:2069:17\n"
    "     |\n"
    "2069 |             let snap = state.clone_solution();\n"
    "     |                 ^^^^ help: if this is intentional, prefix it with an underscore\n"
)
OWN_WARNING = (
    "warning: function `apply_exact_interaction_cluster` is never used\n"
    "   --> tig-algorithms/src/knapsack/talos_cand/track5.rs:769:4\n"
    "    |\n"
    "769 | fn apply_exact_interaction_cluster(state: &mut State) -> bool {\n"
    "    |    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
)
FOREIGN_ERROR = (
    "error[E0308]: mismatched types\n"
    "   --> tig-algorithms/src/lib.rs:9:17\n"
    "    |\n"
    "9   | pub(crate) type HashMap<K, V> = ...;\n"
)
SUMMARY = "warning: `tig-algorithms` (lib) generated 57 warnings\n"
BUILD = ("   Compiling tig-algorithms v0.1.0 (/app/tig-algorithms)\n" + FOREIGN_WARNING + "\n"
         + OWN_WARNING + "\n" + FOREIGN_ERROR + "\n" + SUMMARY
         + "    Finished `release` profile [optimized] target(s) in 3m 35s\n")


def test_relevant_drops_only_warnings_located_in_other_algorithms():
    # mutation: dropping every block with a foreign path loses the error in lib.rs; keeping
    # everything leaves the fix prompt full of other algorithms' warnings
    out = relevant(BUILD)
    assert "superfast_knap_v1" not in out
    assert OWN_WARNING in out
    assert FOREIGN_ERROR in out
    assert SUMMARY in out
    assert "Compiling tig-algorithms" in out and "Finished `release`" in out


def test_relevant_keeps_output_without_diagnostics_verbatim():
    # mutation: a filter that only emits blocks it recognised drops a staging error message
    assert relevant("staging failed: bad path\n") == "staging failed: bad path\n"


def test_dead_new_function_is_reported_when_the_baseline_lacks_it():
    # mutation: ignoring the location reports superfast_knap_v1's dead functions as ours;
    # ignoring the baseline reports the baseline's own dead functions as new
    base = {"track5.rs": ["solve", "old_dead"]}
    out = ("warning: function `old_dead` is never used\n"
           "   --> tig-algorithms/src/knapsack/talos_cand/track5.rs:2:4\n\n"
           "warning: function `build_greedy_density` is never used\n"
           "   --> tig-algorithms/src/knapsack/superfast_knap_v1/track5.rs:198:4\n\n"
           + OWN_WARNING)
    assert dead_new_functions(out, base) == ["track5.rs: apply_exact_interaction_cluster"]


def test_dead_new_method_counts_too():
    # mutation: matching only `function` misses an impl method the agent added
    base = {"track1.rs": []}
    out = ("warning: method `polish` is never used\n"
           "   --> tig-algorithms/src/knapsack/talos_cand/track1.rs:40:8\n")
    assert dead_new_functions(out, base) == ["track1.rs: polish"]


def test_defined_functions_finds_free_functions_methods_and_generics():
    # mutation: a regex without `pub`/indent tolerance misses methods and reports them as new
    text = ("fn a() {}\n  pub fn b(x: i32) {}\nimpl S { pub(crate) fn c<T>(&self) {} }\n"
            "let fnx = 1;\n")
    assert defined_functions(text) == ["a", "b", "c"]
