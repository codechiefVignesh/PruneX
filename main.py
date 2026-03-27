"""
Main code
Interactive mode: user selects which DFA to benchmark.

Options:
  1. HTTP Literal 'GET'
  2. 4-digit sequence
  3. HTTP Methods (GET|POST|PUT|DELETE)
  4. IP Address pattern
  5. Custom string literal (user input)
  0. Exit
"""

import sys
import statistics
from automatons.dfa_builder import (
    make_literal_dfa, make_test_dfa_digit_sequence,
    make_test_dfa_http_pattern, make_test_dfa_ip_address
)
import plots.plot_results as plt
# from automatons.dfa_engine import TraditionalDFAEngine, PrunedDFAEngine
from benchmark.benchmarker import Benchmarker, InputGenerator, build_standard_inputs


WINDOW_SIZE = 16


# ============================================================
# PHASE 1: Correctness Verification
# ============================================================

def phase1_correctness(dfa, dfa_name: str, window_size: int = 16) -> bool:
    print(f"\n{'-'*60}")
    print(f"  PHASE 1: CORRECTNESS — {dfa_name}")
    print(f"{'-'*60}")

    bench = Benchmarker(dfa, window_size)
    errors = []
    tested = 0

    r = bench.run_single("", "empty")
    if not r.correctness_match:
        errors.append(r)
    tested += 1

    for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ':
        r = bench.run_single(c, f"char '{c}'")
        if not r.correctness_match:
            errors.append(r)
        tested += 1

    for length in [10, 50, 100, 500, 1000]:
        for gen_name, gen_fn in [
            ('alpha', InputGenerator.alphabetic_only),
            ('digits', InputGenerator.digits_only),
            ('http', InputGenerator.http_like),
            ('mixed', InputGenerator.mixed_realistic),
            ('random', InputGenerator.fully_random),
        ]:
            for seed in range(5):
                r = bench.run_single(gen_fn(length, seed), f"{gen_name}_{length}")
                if not r.correctness_match:
                    errors.append(r)
                tested += 1

    for pattern in ['GET', 'POST', '1234', '192.168']:
        for length in [100, 500]:
            for gen_fn in [InputGenerator.alphabetic_only, InputGenerator.http_like]:
                inp = InputGenerator.with_embedded_pattern(gen_fn, pattern, length)
                r = bench.run_single(inp, f"embedded_{pattern}")
                if not r.correctness_match:
                    errors.append(r)
                tested += 1

    print(f"  Inputs tested : {tested}")
    if errors:
        print(f"  FAIL — {len(errors)} mismatches")
        for e in errors[:5]:
            print(f"    '{e.input_description}': "
                  f"trad={e.traditional.accepted}, pruned={e.pruned.accepted}")
        return False

    print(f"  PASS — all {tested} inputs agree between both engines.")
    print(f"  No valid match path was removed by pruning.")
    return True


# ============================================================
# PHASE 2: Performance Benchmarking
# ============================================================

def phase2_benchmarks(dfa, dfa_name: str, window_sizes: list = None) -> dict:
    if window_sizes is None:
        window_sizes = [4, 8, 16, 32, 64]

    print(f"\n{'-'*60}")
    print(f"  PHASE 2: BENCHMARKING — {dfa_name}")
    print(f"{'-'*60}")

    all_results = {}

    for ws in window_sizes:
        print(f"\n  Window Size W = {ws}")
        bench = Benchmarker(dfa, window_size=ws)
        input_suites = build_standard_inputs(lengths=[100, 200, 500, 1000])
        suite_summaries = {}

        for suite_name, inputs in input_suites.items():
            suite = bench.run_suite(inputs, f"{suite_name}_W{ws}")
            suite_summaries[suite_name] = suite

            status = '✓' if suite.all_correct else '✗'
            gain_sign = '+' if suite.avg_net_gain >= 0 else ''
            print(f"    {suite_name:<14} "
                  f"correct={status}  "
                  f"n={suite.avg_input_length:.0f}  "
                  f"pruning={suite.avg_pruning_ratio:5.1%}  "
                  f"T_trad={suite.traditional_transitions:.0f}  "
                  f"T_pruned={suite.avg_transitions_pruned:.0f}  "
                  f"net_gain={gain_sign}{suite.avg_net_gain:.0f}  "
                  f"speedup={suite.avg_speedup:.3f}x")

        all_results[ws] = suite_summaries

    return all_results


# ============================================================
# PHASE 3: Formal Proof
# ============================================================

def phase3_proof(dfa, dfa_name: str, window_size: int = 16):
    print(f"\n{'='*60}")
    print(f"  PHASE 3: FORMAL PROOF — {dfa_name}  (W={window_size})")
    print(f"{'='*60}")

    bench = Benchmarker(dfa, window_size)
    input_suites = build_standard_inputs(lengths=[200, 500, 1000])
    # print(len(input_suites))
    all_results = []
    for suite_name, inputs in input_suites.items():
        for inp_str, desc in inputs:
            r = bench.run_single(inp_str, desc)
            all_results.append(r)

    T_trad_avg   = statistics.mean(r.traditional.transitions_evaluated for r in all_results)
    T_prun_avg   = statistics.mean(r.pruned.transitions_evaluated for r in all_results)
    T_prof_avg   = statistics.mean(r.pruned.profiling_cost for r in all_results)
    T_total_avg  = statistics.mean(r.pruned.total_cost for r in all_results)
    net_gain_avg = statistics.mean(r.net_gain for r in all_results)
    pct_positive = sum(1 for r in all_results if r.is_net_positive) / len(all_results) * 100
    correctness_pass = all(r.correctness_match for r in all_results)
    p_avg = 1 - (T_prun_avg / T_trad_avg) if T_trad_avg > 0 else 0

    print(f"""
  Correctness  : {'PASS' if correctness_pass else 'FAIL'} ({len(all_results)} inputs tested)

  Measured values (W = {window_size}, avg over {len(all_results)} inputs):
  ┌─────────────────────────────────────────────────────────┐
  │  T_trad              : {T_trad_avg:>10.1f} ops                   │
  │  T_prun              : {T_prun_avg:>10.1f} ops                   │
  │  T_prof (= 2n)       : {T_prof_avg:>10.1f} ops                   │
  │  T_total             : {T_total_avg:>10.1f} ops                   │
  │  p (pruning ratio)   : {p_avg:>10.1%}                       │
  │  Net gain            : {net_gain_avg:>+10.1f} ops                   │
  │  Net-positive inputs : {pct_positive:>9.1f}%                       │
  └─────────────────────────────────────────────────────────┘

  {'Verification Completed: T_total < T_trad — pruning saves more than it costs.' if net_gain_avg > 0 else 'Net gain negative for this DFA/window combination.'}
""")

    return all_results


# ============================================================
# Run all 3 phases on a given DFA
# ============================================================

def run_all_phases(dfa, name: str):
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"  States={len(dfa.states)}  "
          f"Transitions={dfa.transition_count()}  "
          f"Accept={dfa.accept_states}")
    print(f"{'-'*60}")

    phase1_correctness(dfa, name, WINDOW_SIZE)
    phase2_benchmarks(dfa, name, window_sizes=[4, 8, 16, 32, 64])
    phase3_proof(dfa, name, WINDOW_SIZE)


 
# ============================================================
# Main — interactive menu
# ============================================================

def print_menu():
    print("  Select a DFA to benchmark:")
    print("    1. Literal 'GET'               (KMP substring DFA)")
    print("    2. 4-digit sequence            (Regex .*[0-9]{4})")
    print("    3. HTTP methods                (Regex GET|POST|PUT|DELETE)")
    print("    4. IP address pattern          (Regex IPv4)")
    print("    5. Custom string literal       (you provide the string)")
    print("    6. Run ALL DFAs + generate plots for Analysis")
    print("    0. Exit")
    print("="*60)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  NSD Course Assignment — Problem Statement 5             ║")
    print("║  Input-Aware Transition Pruning in Regex Automata        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    while True:
        print_menu()

        try:
            choice = int(input("  Enter choice: ").strip())
        except ValueError:
            print("  Invalid input. Please enter a number.")
            continue

        if choice == 0:
            print("  Exiting.")
            break

        elif choice == 1:
            dfa  = make_literal_dfa("GET")
            name = "Literal: 'GET'"
            run_all_phases(dfa, name)

        elif choice == 2:
            dfa  = make_test_dfa_digit_sequence()
            name = "Regex: 4-digit sequence"
            run_all_phases(dfa, name)

        elif choice == 3:
            dfa  = make_test_dfa_http_pattern()
            name = "Regex: HTTP methods"
            run_all_phases(dfa, name)

        elif choice == 4:
            dfa  = make_test_dfa_ip_address()
            name = "Regex: IP address"
            run_all_phases(dfa, name)

        elif choice == 5:
            literal = input("  Enter the string literal to search for: ").strip()
            if not literal:
                print("  Empty input — skipping.")
                continue
            dfa  = make_literal_dfa(literal)
            name = f"Literal: '{literal}'"
            run_all_phases(dfa, name)
            
        elif choice == 6:
            plt.run_all_and_plot()
        else:
            print("  Invalid choice. Please enter 0-5.")

        input("\n  Press Enter to return to menu...")

    return 0

if __name__ == "__main__":
    sys.exit(main())