import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from main import run_all_phases, make_literal_dfa, make_test_dfa_digit_sequence, make_test_dfa_http_pattern, make_test_dfa_ip_address
from benchmark.benchmarker import Benchmarker, build_standard_inputs


COLORS = {
    'alphabetic': '#2196F3',
    'http_like':  '#4CAF50',
    'mixed':      '#FF9800',
    'random':     '#F44336',
    'with_matches': '#9C27B0',
}

DFA_COLORS = [
    '#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#AD1457'
]

plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
})

# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def collect_pruning_vs_window(dfa, window_sizes, input_lengths):
    """
    Returns dict: {suite_name: {W: avg_pruning_ratio}}
    """
    data = {}
    for ws in window_sizes:
        bench = Benchmarker(dfa, window_size=ws)
        suites = build_standard_inputs(lengths=input_lengths)
        for suite_name, inputs in suites.items():
            suite = bench.run_suite(inputs, f"{suite_name}_W{ws}")
            if suite_name not in data:
                data[suite_name] = {}
            data[suite_name][ws] = suite.avg_pruning_ratio * 100
    return data


def collect_net_gain_by_dfa(dfa_list, window_sizes, input_lengths):
    """
    Returns dict: {dfa_name: {W: avg_net_gain}} using alphabetic input only.
    """
    data = {}
    for dfa, name in dfa_list:
        data[name] = {}
        for ws in window_sizes:
            bench = Benchmarker(dfa, window_size=ws)
            suites = build_standard_inputs(lengths=input_lengths)
            suite = bench.run_suite(suites['alphabetic'], f"alpha_W{ws}")
            data[name][ws] = suite.avg_net_gain
    return data


def collect_ttrad_vs_ttotal(dfa_list, window_size, input_lengths):
    """
    Returns lists of (name, T_trad, T_prun, T_prof) for a fixed W.
    Uses alphabetic input (best case for pruning).
    """
    rows = []
    for dfa, name in dfa_list:
        bench = Benchmarker(dfa, window_size=window_size)
        suites = build_standard_inputs(lengths=input_lengths)
        suite = bench.run_suite(suites['alphabetic'], f"alpha_W{window_size}")
        rows.append({
            'name': name,
            'T_trad':  suite.traditional_transitions,
            'T_prun':  suite.traditional_transitions - suite.avg_transitions_pruned,
            'T_prof':  suite.avg_profiling_cost,
            'T_saved': suite.avg_transitions_pruned,
        })
    return rows


def collect_speedup_vs_window(dfa_list, window_sizes, input_lengths):
    """
    Returns dict: {dfa_name: {W: avg_speedup}} using alphabetic input.
    """
    data = {}
    for dfa, name in dfa_list:
        data[name] = {}
        for ws in window_sizes:
            bench = Benchmarker(dfa, window_size=ws)
            suites = build_standard_inputs(lengths=input_lengths)
            suite = bench.run_suite(suites['alphabetic'], f"alpha_W{ws}")
            data[name][ws] = suite.avg_speedup
    return data


# ---------------------------------------------------------------------------
# Graph 1: Pruning ratio vs Window size W
# ---------------------------------------------------------------------------

def plot_pruning_vs_window(save_path='plots/graph1_pruning_vs_window.png'):
    print("Collecting data for Graph 1...")
    dfa = make_literal_dfa("Content-Type")
    window_sizes = [4, 8, 16, 32, 64]
    data = collect_pruning_vs_window(dfa, window_sizes, input_lengths=[200, 500, 1000])

    fig, ax = plt.subplots(figsize=(8, 5))

    suite_labels = {
        'alphabetic':   'Alphabetic only',
        'http_like':    'HTTP-like',
        'mixed':        'Mixed realistic',
        'random':       'Fully random',
        'with_matches': 'With embedded match',
    }

    for suite_name, w_dict in data.items():
        ws_list = sorted(w_dict.keys())
        ratios = [w_dict[ws] for ws in ws_list]
        ax.plot(ws_list, ratios,
                marker='o', linewidth=2,
                color=COLORS.get(suite_name, '#333'),
                label=suite_labels.get(suite_name, suite_name))

    ax.set_xlabel('Window Size W')
    ax.set_ylabel('Pruning Ratio (%)')
    ax.set_title('Graph 1: Pruning Ratio vs Window Size\n(Content-Type DFA, avg over n=200,500,1000)')
    ax.set_xticks(window_sizes)
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Graph 2: Net gain vs Window size, grouped by DFA
# ---------------------------------------------------------------------------

def plot_net_gain_by_dfa(dfa_list, save_path='plots/graph2_net_gain_by_dfa.png'):
    print("Collecting data for Graph 2...")
    window_sizes = [8, 16, 32]
    data = collect_net_gain_by_dfa(dfa_list, window_sizes, input_lengths=[200, 500, 1000])

    dfa_names = [name for _, name in dfa_list]
    x = np.arange(len(dfa_names))
    width = 0.25
    offsets = [-width, 0, width]
    w_colors = ['#1565C0', '#2E7D32', '#E65100']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ws in enumerate(window_sizes):
        gains = [data[name][ws] for name in dfa_names]
        bars = ax.bar(x + offsets[i], gains, width,
                      label=f'W={ws}', color=w_colors[i], alpha=0.85)

        for bar, gain in zip(bars, gains):
            if gain > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 200,
                        f'+{gain:.0f}', ha='center', va='bottom',
                        fontsize=7, color=w_colors[i])

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('DFA Pattern')
    ax.set_ylabel('Net Gain (operations saved)')
    ax.set_title('Graph 2: Net Gain by DFA and Window Size\n(alphabetic input, avg over n=200,500,1000)')
    short_names = [n.replace("Literal: '", '').replace("Regex: ", '').rstrip("'")
                   for n in dfa_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Graph 3: T_trad vs T_total (stacked bar — proof visualisation)
# ---------------------------------------------------------------------------

def plot_cost_comparison(dfa_list, save_path='plots/graph3_cost_comparison.png'):
    print("Collecting data for Graph 3...")
    rows = collect_ttrad_vs_ttotal(dfa_list, window_size=16,
                                   input_lengths=[200, 500, 1000])

    names = [r['name'].replace("Literal: '", '').replace("Regex: ", '').rstrip("'")
             for r in rows]
    T_trad  = [r['T_trad'] for r in rows]
    T_prun  = [r['T_prun'] for r in rows]
    T_prof  = [r['T_prof'] for r in rows]
    T_saved = [r['T_saved'] for r in rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Traditional — single solid bar
    ax.bar(x - width/2, T_trad, width, label='T_trad (traditional)',
           color='#B71C1C', alpha=0.85)

    # Pruned total — stacked: T_prun (evaluated) + T_prof (overhead)
    ax.bar(x + width/2, T_prun, width, label='T_prun (evaluated)',
           color='#1565C0', alpha=0.85)
    ax.bar(x + width/2, T_prof, width, bottom=T_prun,
           label='T_prof (profiling overhead)',
           color='#90CAF9', alpha=0.85)

    # Savings annotation
    for i in range(len(names)):
        saving = T_trad[i] - (T_prun[i] + T_prof[i])
        if saving > 0:
            ax.annotate(f'saved\n{saving:.0f}',
                        xy=(x[i] + width/2, T_prun[i] + T_prof[i]),
                        xytext=(x[i] + width/2 + 0.05, T_trad[i] * 0.5),
                        fontsize=7, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

    ax.set_xlabel('DFA Pattern')
    ax.set_ylabel('Total Operations')
    ax.set_title('Graph 3: Traditional vs Pruned Total Cost (W=16)\n'
                 'T_total = T_prun + T_prof  vs  T_trad  (alphabetic input)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Graph 4: Speedup ratio vs Window size
# ---------------------------------------------------------------------------

def plot_speedup_vs_window(dfa_list, save_path='plots/graph4_speedup_vs_window.png'):
    print("Collecting data for Graph 4...")
    window_sizes = [4, 8, 16, 32, 64]
    data = collect_speedup_vs_window(dfa_list, window_sizes,
                                     input_lengths=[200, 500, 1000])

    fig, ax = plt.subplots(figsize=(8, 5))

    for (_, name), color in zip(dfa_list, DFA_COLORS):
        ws_list = sorted(data[name].keys())
        speedups = [data[name][ws] for ws in ws_list]
        short = name.replace("Literal: '", '').replace("Regex: ", '').rstrip("'")
        ax.plot(ws_list, speedups, marker='o', linewidth=2,
                color=color, label=short)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1,
               label='Breakeven (speedup=1)')
    ax.set_xlabel('Window Size W')
    ax.set_ylabel('Speedup Ratio (T_trad / T_total)')
    ax.set_title('Graph 4: Speedup Ratio vs Window Size\n'
                 '(alphabetic input, avg over n=200,500,1000)\n'
                 'Values > 1 mean pruned engine is faster overall')
    ax.set_xticks(window_sizes)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def get_all_dfas():
    return [
        (make_literal_dfa("GET"),           "Literal: 'GET'"),
        (make_literal_dfa("Content-Type"),  "Literal: 'Content-Type'"),
        (make_test_dfa_digit_sequence(),    "Regex: 4-digit sequence"),
        (make_test_dfa_http_pattern(),      "Regex: HTTP methods"),
        (make_test_dfa_ip_address(),        "Regex: IP address"),
    ]

def run_all_and_plot():
    dfa_list = get_all_dfas()
 
    print(f"\n{'-'*60}")
    print(f"  OPTION 6: Run ALL DFAs + Generate Plots")
    print(f"  {len(dfa_list)} DFAs will be benchmarked across all 3 phases.")
    print(f"{'-'*60}")
 
    for dfa, name in dfa_list:
        run_all_phases(dfa, name)
 
    # ---- Now generate the 4 graphs --------------------------------
    print(f"\n{'='*60}")
    print(f"  Generating analytics plots via plot_results.py ...")
    print(f"{'='*60}\n")
 
    try:
        from plots.plot_results import (
            plot_pruning_vs_window,
            plot_net_gain_by_dfa,
            plot_cost_comparison,
            plot_speedup_vs_window,
        )
 
        plot_pruning_vs_window()
        plot_net_gain_by_dfa(dfa_list)
        plot_cost_comparison(dfa_list)
        plot_speedup_vs_window(dfa_list)
 
        print("\n  All graphs saved:")
        print("    graph1_pruning_vs_window.png")
        print("    graph2_net_gain_by_dfa.png")
        print("    graph3_cost_comparison.png")
        print("    graph4_speedup_vs_window.png")
 
    except ImportError as e:
        print(f"  [WARN] Could not import plot_results: {e}")
 