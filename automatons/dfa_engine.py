"""
Two engines are used for DFA execution:

  TraditionalDFAEngine  — evaluates ALL transitions per character
  PrunedDFAEngine       — skips transitions absent from sliding-window profile

Both return:
  run(input_string) -> bool          : whether input is accepted
  transitions_evaluated              : transition lookups performed
  transitions_pruned                 : lookups skipped (pruned engine only)
  profiler.profiling_ops             : overhead ops from window profiler
"""

from collections import deque, Counter
class DFA:
    """
    Explicit DFA as a dict-of-dicts.
      transitions : {state: {char: next_state}}
      Missing entries = implicit dead state.
    """

    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = set(accept_states)

    def transition_count(self):
        return sum(len(v) for v in self.transitions.values())

    def __repr__(self):
        return (f"DFA(states={len(self.states)}, "
                f"alphabet_size={len(self.alphabet)}, "
                f"transitions={self.transition_count()}, "
                f"accept={self.accept_states})")


class TraditionalDFAEngine:
    """
    Standard DFA execution — full transition table model.

    Cost model:
      For each input character, count ALL transitions in the current state.
      (transitions_evaluated += len(delta[state]))
      This reflects real multi-pattern DFA engines where the full row
      of the transition table is consulted per input byte.
    """

    def __init__(self, dfa: DFA):
        self.dfa = dfa
        self.transitions_evaluated = 0

    def reset(self):
        self.transitions_evaluated = 0

    def run(self, input_string: str) -> bool:
        self.reset()
        current = self.dfa.start_state

        for char in input_string:
            state_trans = self.dfa.transitions.get(current, {})
            self.transitions_evaluated += len(state_trans)

            next_state = state_trans.get(char)
            if next_state is None:
                return False
            current = next_state

        return current in self.dfa.accept_states


class SlidingWindowProfiler:
    """
    Tracks Σ'(W) — the set of chars seen in the last W bytes.

    Data structure: deque + frequency Counter.
    All operations are O(1):
      update(char) : add new char, evict oldest when window is full
      active_chars(): return current character set

    Cost: exactly 2 counter operations per character processed.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque()
        self.freq = Counter()
        self.profiling_ops = 0

    def reset(self):
        self.window.clear()
        self.freq.clear()
        self.profiling_ops = 0

    def update(self, char: str):
        self.window.append(char)
        self.freq[char] += 1
        self.profiling_ops += 1

        if len(self.window) > self.window_size:
            evicted = self.window.popleft()
            self.freq[evicted] -= 1
            self.profiling_ops += 1
            if self.freq[evicted] == 0:
                del self.freq[evicted]

    def active_chars(self) -> set:
        return set(self.freq.keys())


class PrunedDFAEngine:
    """
    Input-aware DFA execution with transition pruning.

    Algorithm:
      Phase 1: Pre-compute look-ahead character profiles in O(n).
               Σ'(i) = set of chars appearing in S[i : i+W].
      Phase 2: Execute DFA. At position i, only evaluate transitions
               for chars in Σ'(i); count the rest as pruned.

    Cost model:
      transitions_evaluated += |{c ∈ delta[state] : c ∈ Σ'(i)}|
      transitions_pruned    += |{c ∈ delta[state] : c ∉ Σ'(i)}|
      profiling_ops          = 2n  (two counter ops per character)

    Net gain condition:
      p * T_trad > 2n
      where p = pruning ratio = transitions_pruned / T_trad

    Correctness guarantee:
      delta(q, c) is pruned only when c ∉ S[i:i+W].
      Since position i is included in the window, c cannot be the
      current input character. So no valid transition is ever removed.
    """

    def __init__(self, dfa: DFA, window_size: int):
        self.dfa = dfa
        self.window_size = window_size
        self.profiler = SlidingWindowProfiler(window_size)
        self.transitions_evaluated = 0
        self.transitions_pruned = 0

    def reset(self):
        self.transitions_evaluated = 0
        self.transitions_pruned = 0
        self.profiler.reset()

    def _build_lookahead_profiles(self, input_string: str) -> list:
        """
        Build Σ'(i) for every position using a sliding window — O(n) total.
        Sets profiling_ops = 2n to reflect the cost of this pre-pass.
        """
        n = len(input_string)
        if n == 0:
            return []

        profiles = []
        freq = {}
        right = 0

        for i in range(n):
            while right < n and right < i + self.window_size:
                c = input_string[right]
                freq[c] = freq.get(c, 0) + 1
                right += 1

            profiles.append(set(freq.keys()))

            c = input_string[i]
            freq[c] -= 1
            if freq[c] == 0:
                del freq[c]

        self.profiler.profiling_ops = 2 * n
        return profiles

    def run(self, input_string: str) -> bool:
        self.reset()

        if not input_string:
            return self.dfa.start_state in self.dfa.accept_states

        lookahead = self._build_lookahead_profiles(input_string)
        current = self.dfa.start_state

        for i, char in enumerate(input_string):
            active_set = lookahead[i]
            state_trans = self.dfa.transitions.get(current, {})

            for c in state_trans:
                if c in active_set:
                    self.transitions_evaluated += 1
                else:
                    self.transitions_pruned += 1

            next_state = state_trans.get(char)
            if next_state is None:
                return False
            current = next_state

        return current in self.dfa.accept_states
