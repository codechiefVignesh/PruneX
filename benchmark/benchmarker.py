"""
NSD Problem Statement 5 — Benchmarker
=======================================
Runs TraditionalDFAEngine and PrunedDFAEngine on identical inputs,
collects comparison metrics, and verifies correctness.

Key metrics tracked per input:
  T_trad   = transitions evaluated by traditional engine
  T_prun   = transitions evaluated by pruned engine
  T_prof   = profiling overhead (2 * input_length)
  T_total  = T_prun + T_prof  (true cost of pruned system)
  net_gain = T_trad - T_total (positive = pruning is beneficial)
  pruning_ratio = transitions_pruned / T_trad
"""

import time
import random
import string
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from automatons.dfa_engine import TraditionalDFAEngine, PrunedDFAEngine, DFA


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    accepted: bool
    transitions_evaluated: int
    transitions_pruned: int = 0
    profiling_cost: int = 0
    wall_time_ns: int = 0

    @property
    def total_cost(self):
        return self.transitions_evaluated + self.profiling_cost


@dataclass
class ComparisonResult:
    input_str: str
    input_description: str
    traditional: RunResult
    pruned: RunResult
    correctness_match: bool
    input_length: int = 0

    @property
    def net_gain(self):
        return self.pruned.transitions_pruned - self.pruned.profiling_cost

    @property
    def pruning_ratio(self):
        total = self.traditional.transitions_evaluated
        if total == 0:
            return 0.0
        return self.pruned.transitions_pruned / total

    @property
    def speedup_ratio(self):
        pruned_total = self.pruned.transitions_evaluated + self.pruned.profiling_cost
        if pruned_total == 0:
            return float('inf')
        return self.traditional.transitions_evaluated / pruned_total

    @property
    def is_net_positive(self):
        return self.net_gain > 0


@dataclass
class BenchmarkSuite:
    name: str
    results: List[ComparisonResult] = field(default_factory=list)

    @property
    def all_correct(self):
        return all(r.correctness_match for r in self.results)

    @property
    def avg_pruning_ratio(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.pruning_ratio for r in self.results)

    @property
    def avg_net_gain(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.net_gain for r in self.results)

    @property
    def avg_speedup(self):
        if not self.results:
            return 1.0
        return statistics.mean(r.speedup_ratio for r in self.results)

    @property
    def pct_net_positive(self):
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.is_net_positive) / len(self.results) * 100

    @property
    def traditional_transitions(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.traditional.transitions_evaluated for r in self.results)

    @property
    def avg_transitions_pruned(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.pruned.transitions_pruned for r in self.results)

    @property
    def avg_input_length(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.input_length for r in self.results)

    @property
    def avg_profiling_cost(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.pruned.profiling_cost for r in self.results)

    @property
    def avg_total_cost(self):
        if not self.results:
            return 0.0
        return statistics.mean(r.pruned.total_cost for r in self.results)


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

class InputGenerator:

    @staticmethod
    def alphabetic_only(length: int, seed: int = 42) -> str:
        """Pure letters — no digits or symbols. Best case for digit-pattern DFAs."""
        rng = random.Random(seed)
        return ''.join(rng.choice(string.ascii_letters) for _ in range(length))

    @staticmethod
    def digits_only(length: int, seed: int = 42) -> str:
        """Pure digits."""
        rng = random.Random(seed)
        return ''.join(rng.choice(string.digits) for _ in range(length))

    @staticmethod
    def http_like(length: int, seed: int = 42) -> str:
        """
        HTTP header-like: mostly letters + punctuation, very few digits.
        Simulates realistic structured network input.
        """
        rng = random.Random(seed)
        chars = string.ascii_letters + ': /.-\r\n'
        weights = [2] * len(string.ascii_letters) + [1, 1, 1, 1, 1, 1, 1]
        return ''.join(rng.choices(list(chars), weights=weights)[0] for _ in range(length))

    @staticmethod
    def mixed_realistic(length: int, seed: int = 42) -> str:
        """
        Realistic network payload: ~60% letters, ~20% digits, ~20% symbols.
        Represents average-case input diversity.
        """
        rng = random.Random(seed)
        result = []
        for _ in range(length):
            r = rng.random()
            if r < 0.60:
                result.append(rng.choice(string.ascii_letters))
            elif r < 0.80:
                result.append(rng.choice(string.digits))
            else:
                result.append(rng.choice(' .:/-_@#\r\n'))
        return ''.join(result)

    @staticmethod
    def fully_random(length: int, seed: int = 42) -> str:
        """Uniformly random printable ASCII. Worst case for pruning."""
        rng = random.Random(seed)
        return ''.join(rng.choice(string.printable[:95]) for _ in range(length))

    @staticmethod
    def with_embedded_pattern(base_fn, pattern: str,
                               length: int = 500, seed: int = 42) -> str:
        """Embed a matching pattern into the middle of a base string."""
        base = base_fn(length, seed)
        pos = length // 2
        return base[:pos] + pattern + base[pos:]


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class Benchmarker:

    def __init__(self, dfa: DFA, window_size: int = 16):
        self.dfa = dfa
        self.window_size = window_size
        self.trad_engine = TraditionalDFAEngine(dfa)
        self.prun_engine = PrunedDFAEngine(dfa, window_size)

    def run_single(self, input_str: str, description: str = "") -> ComparisonResult:
        actual_length = len(input_str)

        t0 = time.perf_counter_ns()
        trad_accepted = self.trad_engine.run(input_str)
        t1 = time.perf_counter_ns()
        trad_result = RunResult(
            accepted=trad_accepted,
            transitions_evaluated=self.trad_engine.transitions_evaluated,
            wall_time_ns=t1 - t0
        )

        t2 = time.perf_counter_ns()
        prun_accepted = self.prun_engine.run(input_str)
        t3 = time.perf_counter_ns()
        prun_result = RunResult(
            accepted=prun_accepted,
            transitions_evaluated=self.prun_engine.transitions_evaluated,
            transitions_pruned=self.prun_engine.transitions_pruned,
            profiling_cost=self.prun_engine.profiler.profiling_ops,
            wall_time_ns=t3 - t2
        )

        return ComparisonResult(
            input_str=input_str[:50] + ('...' if len(input_str) > 50 else ''),
            input_description=description,
            traditional=trad_result,
            pruned=prun_result,
            correctness_match=(trad_accepted == prun_accepted),
            input_length=actual_length
        )

    def run_suite(self, inputs: List[Tuple[str, str]], suite_name: str) -> BenchmarkSuite:
        suite = BenchmarkSuite(name=suite_name)
        for input_str, desc in inputs:
            suite.results.append(self.run_single(input_str, desc))
        return suite


# ---------------------------------------------------------------------------
# Standard input suites
# ---------------------------------------------------------------------------

def build_standard_inputs(lengths: List[int] = None) -> Dict[str, List[Tuple[str, str]]]:
    if lengths is None:
        lengths = [100, 200, 500, 1000]

    suites = {}

    suites['alphabetic'] = [
        (InputGenerator.alphabetic_only(n, seed=i), f"alpha_n={n}_s={i}")
        for n in lengths for i in range(5)
    ]
    suites['http_like'] = [
        (InputGenerator.http_like(n, seed=i), f"http_n={n}_s={i}")
        for n in lengths for i in range(5)
    ]
    suites['mixed'] = [
        (InputGenerator.mixed_realistic(n, seed=i), f"mixed_n={n}_s={i}")
        for n in lengths for i in range(5)
    ]
    suites['random'] = [
        (InputGenerator.fully_random(n, seed=i), f"rand_n={n}_s={i}")
        for n in lengths for i in range(5)
    ]
    suites['with_matches'] = [
        (InputGenerator.with_embedded_pattern(InputGenerator.alphabetic_only, '1234', n, seed=i),
         f"match_n={n}_s={i}")
        for n in lengths for i in range(5)
    ]

    return suites
