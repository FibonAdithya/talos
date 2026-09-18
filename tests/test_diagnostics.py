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


def test_dead_new_methods_grouped_in_one_warning_are_all_reported():
    # rustc groups unused methods of one impl: the run's logs carry 8 lines of the form
    # "methods `a`, `b`, and `c` are never used". mutation: matching only the singular form
    # scores a candidate whose new impl is dead in full
    base = {"track1.rs": ["solve", "old"]}
    out = ("warning: methods `old`, `relax`, and `polish` are never used\n"
           "   --> tig-algorithms/src/knapsack/talos_cand/track1.rs:40:8\n\n"
           "warning: functions `a` and `b` are never used\n"
           "   --> tig-algorithms/src/knapsack/superfast_knap_v1/track1.rs:1:1\n\n"
           "warning: associated functions `mk` and `go` are never used\n"
           "   --> tig-algorithms/src/knapsack/talos_cand/track1.rs:80:8\n")
    assert dead_new_functions(out, base) == ["track1.rs: relax", "track1.rs: polish",
                                            "track1.rs: mk", "track1.rs: go"]


def test_first_error_is_the_first_error_line_not_cargos_summary():
    # mutation: taking the last line prints cargo's summary ("could not compile ... due to 2
    # previous errors") whenever a real error exists; taking the first line prints a warning
    from talos.diagnostics import first_error
    output = ("warning: unused variable: `hp`\n   --> src/a.rs:1:1\n"
              "error[E0382]: borrow of moved value: `order`\n    --> src/t.rs:9:5\n"
              "error[E0004]: non-exhaustive patterns\n"
              "error: could not compile `tig-algorithms` (lib) due to 2 previous errors\n")
    assert first_error(output) == "error[E0382]: borrow of moved value: `order`"
    assert first_error("  error: expected `;`\nerror: aborting due to 1 previous error\n") == \
        "error: expected `;`"
    # only the summary survives: better than nothing
    assert first_error("warning: x\nerror: could not compile `t` (lib) due to 2 previous errors") \
        == "error: could not compile `t` (lib) due to 2 previous errors"
    # no error line at all (a linker message, a timeout note): the last non-empty line
    assert first_error("note: a\nkilled after 600s\n\n") == "killed after 600s"
    assert first_error("") == ""
