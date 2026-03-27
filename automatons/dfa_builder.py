"""
DFA Builder
======================================
Converts regular expressions into explicit DFA transition tables
using Thompson's NFA construction followed by subset (powerset) construction.

Supported regex constructs:
  Literals, concatenation, alternation (|), Kleene star (*),
  one-or-more (+), zero-or-one (?), grouping (()), character classes ([]),
  dot (.), escape sequences (\\d \\w \\s).

Pre-built test DFAs:
  make_literal_dfa(s)          — substring search via KMP-style DFA
  make_test_dfa_digit_sequence — matches strings containing 4 consecutive digits
  make_test_dfa_http_pattern   — matches strings containing GET/POST/PUT/DELETE
  make_test_dfa_ip_address     — matches strings containing an IPv4 address
"""

from collections import defaultdict, deque
from automatons.dfa_engine import DFA


PRINTABLE_ASCII = set(chr(c) for c in range(32, 127))


# NFA State

class NFAState:
    _id_counter = 0

    def __init__(self):
        NFAState._id_counter += 1
        self.id = NFAState._id_counter
        self.transitions = defaultdict(set)
        self.epsilon = set()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f"S{self.id}"


class NFA:
    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept



# Character class parser and escape handler

def _parse_char_class(pattern: str, pos: int):
    """Parse [abc], [a-z], [^0-9] starting at pos. Returns (char_set, new_pos)."""
    assert pattern[pos] == '['
    pos += 1
    negate = False

    if pos < len(pattern) and pattern[pos] == '^':
        negate = True
        pos += 1

    chars = set()
    while pos < len(pattern) and pattern[pos] != ']':
        if pos + 2 < len(pattern) and pattern[pos+1] == '-' and pattern[pos+2] != ']':
            for c in range(ord(pattern[pos]), ord(pattern[pos+2]) + 1):
                chars.add(chr(c))
            pos += 3
        elif pattern[pos] == '\\' and pos + 1 < len(pattern):
            pos += 1
            result = _escape(pattern[pos])
            if isinstance(result, set):
                chars.update(result)
            else:
                chars.add(result)
            pos += 1
        else:
            chars.add(pattern[pos])
            pos += 1

    pos += 1  # consume ']'
    if negate:
        chars = PRINTABLE_ASCII - chars
    return chars, pos


def _escape(c: str):
    mapping = {
        'd': set('0123456789'),
        'w': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'),
        's': set(' \t\n\r'),
        'n': '\n', 't': '\t', 'r': '\r'
    }
    return mapping.get(c, c)


# Recursive descent regex parser → NFA (Thompson's construction)

class RegexParser:
    """
    Grammar:
      expr   := term ('|' term)*
      term   := factor+
      factor := atom ('*' | '+' | '?')?
      atom   := char | '.' | '(' expr ')' | '[' class ']' | '\\' escape
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    def parse(self) -> NFA:
        return self._expr()

    def _expr(self) -> NFA:
        left = self._term()
        while self.pos < len(self.pattern) and self.pattern[self.pos] == '|':
            self.pos += 1
            right = self._term()
            left = self._alternate(left, right)
        return left

    def _term(self) -> NFA:
        factors = []
        while self.pos < len(self.pattern) and self.pattern[self.pos] not in '|)':
            factors.append(self._factor())
        if not factors:
            s, a = NFAState(), NFAState()
            s.epsilon.add(a)
            return NFA(s, a)
        result = factors[0]
        for f in factors[1:]:
            result = self._concat(result, f)
        return result

    def _factor(self) -> NFA:
        atom = self._atom()
        if self.pos < len(self.pattern):
            q = self.pattern[self.pos]
            if q == '*':
                self.pos += 1
                return self._kleene_star(atom)
            elif q == '+':
                self.pos += 1
                return self._one_or_more(atom)
            elif q == '?':
                self.pos += 1
                return self._zero_or_one(atom)
        return atom

    def _atom(self) -> NFA:
        if self.pos >= len(self.pattern):
            s, a = NFAState(), NFAState()
            s.epsilon.add(a)
            return NFA(s, a)

        c = self.pattern[self.pos]

        if c == '(':
            self.pos += 1
            nfa = self._expr()
            assert self.pattern[self.pos] == ')'
            self.pos += 1
            return nfa
        elif c == '[':
            char_set, self.pos = _parse_char_class(self.pattern, self.pos)
            return self._char_set_nfa(char_set)
        elif c == '.':
            self.pos += 1
            return self._char_set_nfa(PRINTABLE_ASCII)
        elif c == '\\':
            self.pos += 1
            esc = self.pattern[self.pos]
            self.pos += 1
            result = _escape(esc)
            if isinstance(result, set):
                return self._char_set_nfa(result)
            return self._single_char_nfa(result)
        else:
            self.pos += 1
            return self._single_char_nfa(c)

    def _single_char_nfa(self, char: str) -> NFA:
        s, a = NFAState(), NFAState()
        s.transitions[char].add(a)
        return NFA(s, a)

    def _char_set_nfa(self, chars: set) -> NFA:
        s, a = NFAState(), NFAState()
        for ch in chars:
            s.transitions[ch].add(a)
        return NFA(s, a)

    def _concat(self, n1: NFA, n2: NFA) -> NFA:
        n1.accept.epsilon.add(n2.start)
        return NFA(n1.start, n2.accept)

    def _alternate(self, n1: NFA, n2: NFA) -> NFA:
        s, a = NFAState(), NFAState()
        s.epsilon.add(n1.start)
        s.epsilon.add(n2.start)
        n1.accept.epsilon.add(a)
        n2.accept.epsilon.add(a)
        return NFA(s, a)

    def _kleene_star(self, n: NFA) -> NFA:
        s, a = NFAState(), NFAState()
        s.epsilon.add(n.start)
        s.epsilon.add(a)
        n.accept.epsilon.add(n.start)
        n.accept.epsilon.add(a)
        return NFA(s, a)

    def _one_or_more(self, n: NFA) -> NFA:
        s, a = NFAState(), NFAState()
        s.epsilon.add(n.start)
        n.accept.epsilon.add(n.start)
        n.accept.epsilon.add(a)
        return NFA(s, a)

    def _zero_or_one(self, n: NFA) -> NFA:
        s, a = NFAState(), NFAState()
        s.epsilon.add(n.start)
        s.epsilon.add(a)
        n.accept.epsilon.add(a)
        return NFA(s, a)


# NFA → DFA (subset construction)

def _epsilon_closure(states: frozenset) -> frozenset:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


def _move(states: frozenset, char: str) -> frozenset:
    result = set()
    for s in states:
        result.update(s.transitions.get(char, set()))
    return frozenset(result)


def nfa_to_dfa(nfa: NFA, alphabet: set) -> DFA:
    start_closure = _epsilon_closure(frozenset([nfa.start]))
    dfa_states = {start_closure: 0}
    state_counter = [0]
    queue = deque([start_closure])
    transitions = defaultdict(dict)
    accept_states = set()

    if nfa.accept in start_closure:
        accept_states.add(0)

    while queue:
        current_set = queue.popleft()
        current_id = dfa_states[current_set]

        for char in alphabet:
            moved = _move(current_set, char)
            if not moved:
                continue
            next_closure = _epsilon_closure(moved)

            if next_closure not in dfa_states:
                state_counter[0] += 1
                new_id = state_counter[0]
                dfa_states[next_closure] = new_id
                queue.append(next_closure)
                if nfa.accept in next_closure:
                    accept_states.add(new_id)

            transitions[current_id][char] = dfa_states[next_closure]

    return DFA(
        states=set(dfa_states.values()),
        alphabet=alphabet,
        transitions=dict(transitions),
        start_state=0,
        accept_states=accept_states
    )


def regex_to_dfa(pattern: str, alphabet: set = None) -> DFA:
    if alphabet is None:
        alphabet = PRINTABLE_ASCII
    NFAState._id_counter = 0
    nfa = RegexParser(pattern).parse()
    return nfa_to_dfa(nfa, alphabet)


# Pre-built test DFAs


def make_literal_dfa(literal: str) -> DFA:
    """
    DFA accepting any string containing 'literal' as a substring.
    Built using KMP failure function for correct fallback transitions.
    Alphabet: full printable ASCII (95 chars).
    """
    n = len(literal)
    alphabet = PRINTABLE_ASCII
    transitions = {}
    accept_states = {n}

    fail = [0] * (n + 1)
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and literal[i] != literal[j]:
            j = fail[j - 1]
        if literal[i] == literal[j]:
            j += 1
        fail[i] = j

    for state in range(n + 1):
        transitions[state] = {}
        for c in alphabet:
            if state < n and c == literal[state]:
                transitions[state][c] = state + 1
            else:
                j = fail[state - 1] if state > 0 else 0
                while j > 0 and c != literal[j]:
                    j = fail[j - 1]
                if c == literal[j]:
                    j += 1
                transitions[state][c] = j

    return DFA(
        states=set(range(n + 2)),
        alphabet=alphabet,
        transitions=transitions,
        start_state=0,
        accept_states=accept_states
    )


def make_test_dfa_digit_sequence() -> DFA:
    """
    Accepts strings containing 4 consecutive digits anywhere.
    Uses full printable ASCII alphabet — digit transitions are sparse
    relative to the alphabet, creating strong pruning on alphabetic input.
    """
    return regex_to_dfa(r'.*[0-9][0-9][0-9][0-9]', alphabet=PRINTABLE_ASCII)


def make_test_dfa_http_pattern() -> DFA:
    """
    Accepts strings containing GET, POST, PUT, or DELETE anywhere.
    Alphabet: letters + digits + space + slash (64 chars).
    """
    return regex_to_dfa(
        r'.*(GET|POST|PUT|DELETE)',
        alphabet=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                     'abcdefghijklmnopqrstuvwxyz'
                     '0123456789 /')
    )


def make_test_dfa_ip_address() -> DFA:
    """
    Accepts strings containing an IPv4 address (e.g. 192.168.1.1) anywhere.
    Alphabet: digits + letters + dot (63 chars).
    Digit-heavy pattern — alphabetic input triggers heavy pruning.
    """
    return regex_to_dfa(
        r'.*[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+',
        alphabet=set('0123456789'
                     'abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                     '.')
    )
