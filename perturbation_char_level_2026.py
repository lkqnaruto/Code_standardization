import random
import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import logging
import math
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import string   
import re
# --- helpers 
_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"          # handles hyphenated words & apostrophes: policy-maker, bank's
_PLACEHOLDER = r"__PHRASE_\d+__"
_PUNCT = r"[^\w\s]"                             # any single non-word, non-space (.,;:!?()[]{}"”’ etc.)
_NUMBER = r"\d+(?:[.,]\d+)*%?|\$\d+(?:[.,]\d+)*"
TOKEN_RX = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_PUNCT}|{_NUMBER}")

def tokenize(text: str):
    # Returns a list of tokens: words/placeholders/punctuation
    return TOKEN_RX.finditer(text)



def _choose_positions(s: str, 
                      n_edits: int, 
                      max_per_word: int = 2,
                      boundary_skip_p: float = 0.8,
                      perturbation_type: str = "deletion") -> List[int]:
    """
    Select up to n_edits character positions in s to perturb.
    No word contributes more than max_per_word positions.

    boundary_skip_p: probability to skip a boundary character within a word (start/end index).
    """
    if n_edits <= 0 or max_per_word <= 0:
        return []

    # 1) Find non-space "words" as spans (handles multiple spaces/tabs cleanly)

    spans = []  # (start, end, tok)
    for m in TOKEN_RX.finditer(s):
        start, end = m.span()
        tok = s[start:end]
        spans.append((start, end, tok))

    # 2) Collect candidate positions per word
    per_word_positions = []  # list[list[int]]
    for start, end, tok in spans:
        if end <= start:
            per_word_positions.append([])
            continue

        if  _is_mostly_numeric(tok, level = 0.6):
            per_word_positions.append([])
            continue
        if is_punctuation(tok):
            per_word_positions.append([])
            continue
        
        # Helper: is deleting idx going to zero out its token?
        def _would_zero_token(idx: int, start_idx: int, end_idx: int) -> bool:
            if start_idx <= idx < end_idx:
                # token length after deletion
                new_len = (end_idx - start_idx) - 1
                return new_len <= 0
            return False  # idx not in a token (e.g., whitespace between tokens)


        positions = []
        for i in range(start, end):
            ch = s[i]
            if ch.isspace():
                continue
            if _would_zero_token(i, start, end) and perturbation_type == "deletion":
                continue
            if i < end-1:
                if s[i] == s[i+1]:
                    continue

            at_boundary = (i == start) or (i == end - 1)

            # Optionally downweight very short tokens
            if len(tok) <= 2 and random.random() < 0.8:
                continue

            # Optionally skip boundaries
            if at_boundary and random.random() < boundary_skip_p:
                continue

            positions.append(i)

        # 3) Shuffle and hard-cap per word to avoid oversampling later
        if positions:
            random.shuffle(positions)
            per_word_positions.append(positions[:max_per_word])
        else:
            per_word_positions.append([])

    random.shuffle(per_word_positions)
    # 4) Round-robin selection for fairness, respecting global n_edits
    total_available = sum(len(lst) for lst in per_word_positions)
    target = min(n_edits, total_available)

    selected = []
    round_idx = 0
    while len(selected) < target:
        progressed = False
        for lst in per_word_positions:
            if round_idx < len(lst):
                selected.append(lst[round_idx])
                if len(selected) >= target:
                    break
                progressed = True
        if not progressed:
            break
        round_idx += 1

    # Positions are unique by construction; return as-is
    return selected

def _is_mostly_numeric(tok: str, level: float = 0.6) -> bool:
    digits = sum(c.isdigit() for c in tok)
    return digits >= level * max(1, len(tok))


def is_punctuation(token: str) -> bool:
    """Return True if the token is a punctuation character."""
    import string
    # return token in string.punctuation
    return all(ch in string.punctuation for ch in token)


def solve_a_b(n1, n2, d1, d2):

    # Solve for a, b from two anchors for Δ(n) = (a + b ln n)/n
    b = (d1*n1 - d2*n2) / (np.log(n1) - np.log(n2))
    a = d1*n1 - b*np.log(n1)
    return a, b


def cer_estimation(n, i, n0, level="Conservative"):
    mu, sigma = 0.0117, 0.0143 
    scenarios = {
        "Conservative": {"Delta1": 0.050, "Delta2": 0.010},
        "Balanced":     {"Delta1": 0.010, "Delta2": 0.012},
        "Aggressive":   {"Delta1": 0.015, "Delta2": 0.015},
    }
    d1, d2 = scenarios[level]["Delta1"], scenarios[level]["Delta2"]
    a, b = solve_a_b(20, 40, d1, d2)
    # Option B: CER(n|i) = mu + i*sigma + i * (a + b ln n) / n
    return  mu + i*sigma + i * (a + b*np.log(n+n0)) / n



def _expected_edits(n: int, intensity: str) -> float:
    if n <= 1:
        return 0.0
    # Ensure expected edits never decrease as n increases
    n0: int = 5
    mult = {"low": 1.0, "moderate": 2.0, "high": 3.0}.get(intensity, 1.0)
    

    lam = cer_estimation(n, mult, n0) * n
    
    # B: float = 1.5
    # n0: int = 30
    # max_cer: float = 0.05
    # # Core curve: sublinear growth; damped for ultra-short by n0
    # lam = mult * B * math.log(1.0 + n / float(n0))
    return lam

def _integerize_edits(lam: float, mode: str = "stochastic") -> int:
        # Integerize
    if mode == "poisson":
        # Knuth sampler (small lambdas); replace with numpy if available
        if lam <= 0: k = 0
        else:
            L = math.exp(-lam)
            k, p = 0, 1.0
            while p > L:
                k += 1
                p *= random.random()
            k = max(0, k - 1)
    else:
        base = math.floor(lam)
        frac = lam - base
        k = base + (1 if random.random() < frac else 0)
    return k

# --- 3) insertion char from context ---
def _insert_char_for_gap(s: str, # text
                         g: int, # idx
                         typo_map: dict) -> str:
    """Pick an inserted char using local context + typo_map when possible."""

    left  = s[g-1] if g-1 >= 0 else ''
    right = s[g] if g < len(s) else ''
    # Prioritize: duplicate left, else right, else 'e'
    if random.random() < 0.8:
        if left: return left
        if right: return right
    else:
        base  = left if left else right
        lower = base.lower() if base else None

        if lower and typo_map.get(lower):
            c = random.choice(typo_map[lower])
            return c.upper() if (left and left.isupper()) else c

    return 'e'



import numpy as np
import random
import string
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

class PerturbationType(Enum):
    """Types of character-level perturbations"""
    TYPO = "typo"
    DELETION = "deletion"
    INSERTION = "insertion"
    TRANSPOSITION = "transposition"

@dataclass
class TestCase:
    """Represents a single test case"""
    original_query: str
    perturbed_query: str
    perturbation_type: PerturbationType
    intensity: str

class CharacterPerturbator:
    """Handles character-level perturbations with adjustable intensity"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common typo mappings
        self.typo_map = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'n', 'g'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'c'],
            'e': ['w', 'r', '3'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', '8'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'p', '1'],
            'm': ['n'],
            'n': ['b', 'm', 'h'],
            'o': ['i', 'p', '0'],
            'p': ['o', 'l', '0'],
            'q': ['w', 'a', '1'],
            'r': ['e', 't', '4'],
            's': ['a', 'd', 'z'],
            't': ['r', 'y', '5'],
            'u': ['y', 'i', '7'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', '2'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', '6'],
            'z': ['a', 'x', 's']
        }
        
        # Unicode lookalikes
        self.unicode_map = {
            'a': ['а', 'ɑ', 'α'],
            'e': ['е', 'ε', 'ɛ'],
            'i': ['і', 'ι', 'ɪ'],
            'o': ['о', 'ο', 'σ'],
            'c': ['с', 'ϲ'],
            'p': ['р', 'ρ'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
        }
    
    def apply_perturbation(self, 
                           text: str,
                           perturbation_type: PerturbationType, 
                           intensity: float) -> str:
        """Apply a specific perturbation type with given intensity"""
        if perturbation_type == PerturbationType.TYPO:
            return self._apply_typo(text, intensity)
        elif perturbation_type == PerturbationType.DELETION:
            return self._apply_deletion(text, intensity)
        elif perturbation_type == PerturbationType.INSERTION:
            return self._apply_insertion(text, intensity)
        elif perturbation_type == PerturbationType.TRANSPOSITION:
            return self._apply_transposition(text, intensity)

        else:
            return text
    

    def _apply_typo(self, text: str, intensity: str, mode: str = "stochastic") -> str:
        """Simulate keyboard typos with length-aware intensity and realism."""
        if not text:
            return text

        # ---- tunables (adjust to taste) ----
        max_cer = 0.08       # hard cap on edits as a fraction of length
        min_len_for_forced_edit = 5  # don't force edits for very short strings
        boundary_skip_p = 0.8 # avoid first/last char of a word ~70% of the time
        # ------------------------------------

        n_chars = len(text) 

        # Mutate a single character using typo_map; preserve case
        def _mutate_char(orig: str) -> str:
            lower = orig.lower()
            if lower in getattr(self, "typo_map", {}) and self.typo_map[lower]:
                repl = random.choice(self.typo_map[lower])
                return repl.upper() if orig.isupper() else repl
            return orig  # no mapping -> leave unchanged

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))

        edits = min(edits, max_edits)
  
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0

        if edits <= 0:
            return text

        # 4) pick positions and apply edits
        positions = _choose_positions(text, edits)
        if not positions:
            return text

        chars = list(text)
        for idx in positions:
            chars[idx] = _mutate_char(chars[idx])

        return ''.join(chars)



    def _apply_deletion(self, text: str, 
                        intensity: str, 
                        max_cer: float = 0.08) -> str:

        min_len_for_forced_edit = 10

        n_chars = len(text)

        lam = _expected_edits(n_chars, intensity)

        edits = _integerize_edits(lam)

        max_edits = max(1, int(n_chars * max_cer))

        edits = min(edits, max_edits)

        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0
     
        if edits <= 0:
            return text

        positions = _choose_positions(text, edits, boundary_skip_p=0.9)
        chars = list(text)
        for idx in sorted(positions, reverse=True):  # prevent index shift
            if 0 <= idx < len(chars):
                del chars[idx]
        out = ''.join(chars)

        return out


    def _apply_insertion(self, 
                         text: str, 
                         intensity: str, 
                         mode: str = "stochastic") -> str:
        """Insert realistic characters (adjacent on keyboard)."""
        if not text:
            return text
        min_len_for_forced_edit = 10
        max_cer = 0.08       # hard cap on edits as a fraction of length

        n_chars = len(text)

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))
        edits = min(edits, max_edits)
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0

        if edits <= 0:
            return text

        # 4) pick positions and apply edits
        positions = _choose_positions(text, edits, boundary_skip_p=1.0)
        if not positions:
            return text
        
        chars = list(text)
        for idx in sorted(positions, reverse=True):  # prevent index shift
            if 0 <= idx < len(chars):
                insert_char = _insert_char_for_gap(text, idx, self.typo_map)
                chars.insert(idx, insert_char)

        return ''.join(chars)


    
    def _apply_transposition(self, 
                             text: str, 
                             intensity: str,
                             mode: str = "stochastic",
                             boundary_skip_p: float = 0.9) -> str:
        """Transpose adjacent characters."""
        if not text:
            return text
        min_len_for_forced_edit = 4
        max_cer = 0.08       # hard cap on edits as a fraction of length
        max_per_word: int = 2  # max transpositions per word
        n_chars = len(text)

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))
        edits = min(edits, max_edits)
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.8 else 0

        if edits <= 0:
            return text


        spans = []  # (start, end, tok)
        for m in TOKEN_RX.finditer(text):
            start, end = m.span()
            tok = text[start:end]
            spans.append((start, end, tok))

        per_word_pairs = []  # each entry: list of valid start indices i to swap (i,i+1)
        for start, end, tok in spans:
            if end <= start:
               per_word_pairs.append([])
               continue
            # If token is mostly digits, always allow transposition
            if _is_mostly_numeric(tok, level=0.6):
                starts = []
                for i in range(start, end - 1):
                    if text[i].isspace() or text[i+1].isspace():
                        continue
                    starts.append(i)
                random.shuffle(starts)
                chosen = []
                blocked = set()
                for i in starts:
                    if i in blocked or (i - 1) in blocked or (i + 1) in blocked:
                        continue
                    chosen.append(i)
                    blocked.update({i - 1, i, i + 1})
                    if len(chosen) >= max_per_word:
                        break
                per_word_pairs.append(chosen)
                continue


            if end - start < 2 or is_punctuation(tok):
                per_word_pairs.append([])
                continue

            starts = []
            for i in range(start, end - 1):
                # avoid whitespace & boundaries
                if text[i].isspace() or text[i+1].isspace():
                    continue
                at_left_boundary  = (i == start)
                at_right_boundary = (i + 1 == end - 1)
                if (at_left_boundary or at_right_boundary) and random.random() < boundary_skip_p:
                    continue
                # skip identical pair to increase visible effect, e.g., "ee"
                if text[i] == text[i+1]:
                    continue
                starts.append(i)

            random.shuffle(starts)
            # take up to max_per_word *non-overlapping* starts
            chosen = []
            blocked = set()
            for i in starts:
                if i in blocked or (i - 1) in blocked or (i + 1) in blocked:
                    continue
                chosen.append(i)
                # block neighbors so pairs don't overlap (i,i+1) conflicts with i-1 and i+1
                blocked.update({i - 1, i, i + 1})
                if len(chosen) >= max_per_word:
                    break

            per_word_pairs.append(chosen)
        
        random.shuffle(per_word_pairs)
        total_avail = sum(len(lst) for lst in per_word_pairs)
        target = min(edits, total_avail)
        selected, rr = [], 0
        while len(selected) < target:
            progressed = False
            for lst in per_word_pairs:
                if rr < len(lst):
                    selected.append(lst[rr])
                    if len(selected) >= target:
                        break
                    progressed = True
            if not progressed:
                break
            rr += 1

        if not selected:
            return text
        # ----- apply swaps -----
        # Swapping adjacent chars does not change length → no index shift issues.
        chars = list(text)
        # Sort ascending just for determinism; overlaps already prevented
        # print("Selected positions for transposition:", selected)
        for i in sorted(selected):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return ''.join(chars)
    
    

def adapt_model_interface(
    api_fn: Callable,
    *,
    response_key: Optional[str] = None,
    id_field: str = "id",
    score_field: str = "score",
    sort: bool = True,
    top_k: int = 10,
    extra_kwargs: Optional[Dict] = None,
) -> Callable[[str], List[Tuple[int, float]]]:
    """Convert an arbitrary scoring function / API caller into the
    ``query -> List[Tuple[doc_id, score]]`` interface expected by
    :class:`HybridIRModelTester`.

    Supported *api_fn* return shapes (auto-detected):

    1. **Already compatible** – ``List[Tuple[int, float]]``
    2. **List of dicts**     – ``[{"id": 3, "score": 0.91}, ...]``
       Use *id_field* / *score_field* to match your API's key names.
    3. **Dict with a results key** – ``{"hits": [...], ...}``
       Set *response_key* (e.g. ``"hits"``) so the converter can
       drill into the response first, then apply rule 2.
    4. **Dict mapping id→score** – ``{0: 0.8, 3: 0.71, ...}``
    5. **Flat score array / list** – ``[0.1, 0.4, 0.9, ...]``
       Index position is treated as doc_id.

    Parameters
    ----------
    api_fn : Callable
        The raw scoring function.  Must accept at least a *query*
        positional argument.
    response_key : str, optional
        If the function returns a dict, extract this key first.
    id_field / score_field : str
        Key names when results are dicts.
    sort : bool
        If True, sort descending by score and keep *top_k*.
    top_k : int
        Maximum number of results to return.
    extra_kwargs : dict, optional
        Additional keyword arguments forwarded to *api_fn* on every call.

    Returns
    -------
    Callable[[str], List[Tuple[int, float]]]
        A function compatible with ``HybridIRModelTester(model_interface=...)``.

    Example
    -------
    >>> def my_api(query, n=10):
    ...     resp = requests.post(URL, json={"q": query, "k": n})
    ...     return resp.json()          # {"hits": [{"_id": 5, "_score": 0.9}, ...]}
    >>> adapted = adapt_model_interface(
    ...     my_api,
    ...     response_key="hits",
    ...     id_field="_id",
    ...     score_field="_score",
    ...     extra_kwargs={"n": 20},
    ... )
    >>> tester = HybridIRModelTester(model_interface=adapted)
    """

    _extra = extra_kwargs or {}

    def _convert(raw) -> List[Tuple[int, float]]:
        # --- Step 1: unwrap response_key if needed ---
        if response_key is not None and isinstance(raw, dict):
            raw = raw[response_key]

        # --- Step 2: detect shape & normalise ---
        if isinstance(raw, list) and len(raw) == 0:
            return []

        # 2a  List[Tuple | List]  – already (id, score) pairs?
        if isinstance(raw, list) and isinstance(raw[0], (tuple, list)):
            pairs = [(int(r[0]), float(r[1])) for r in raw]

        # 2b  List[dict]
        elif isinstance(raw, list) and isinstance(raw[0], dict):
            pairs = [
                (int(r[id_field]), float(r[score_field])) for r in raw
            ]

        # 2c  Dict  id→score
        elif isinstance(raw, dict):
            pairs = [(int(k), float(v)) for k, v in raw.items()]

        # 2d  ndarray / flat list of scores  → index = doc_id
        elif isinstance(raw, (np.ndarray, list)):
            arr = np.asarray(raw, dtype=float)
            pairs = [(int(i), float(s)) for i, s in enumerate(arr)]

        else:
            raise TypeError(
                f"adapt_model_interface: cannot convert return type "
                f"{type(raw).__name__}.  Supported: list[tuple], "
                f"list[dict], dict, ndarray."
            )

        # --- Step 3: sort & trim ---
        if sort:
            pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def _adapted(query: str) -> List[Tuple[int, float]]:
        result = api_fn(query, **_extra)
        return _convert(result)

    # Preserve original docstring for introspection
    _adapted.__doc__ = (
        f"Adapted interface around {getattr(api_fn, '__name__', repr(api_fn))}.\n"
        f"Original doc: {getattr(api_fn, '__doc__', 'N/A')}"
    )
    _adapted.__wrapped__ = api_fn
    return _adapted


def _rbo(list1: List, list2: List, p: float = 0.9) -> float:
    """Rank-Biased Overlap between two ranked lists.

    Designed for partial / truncated rankings (e.g. top-3 or top-10).
    *p* controls depth weighting: higher values give more weight to lower
    ranks.  Typical values: 0.9 (top-10 focus), 0.98 (top-50 focus).
    Returns a value in [0, 1]; 1.0 means identical ordering.
    """
    s, t = list(list1), list(list2)
    sl, tl = len(s), len(t)
    if sl == 0 and tl == 0:
        return 1.0
    max_depth = max(sl, tl)
    # RBO with extrapolated residual for short / truncated lists
    x_d = 0.0        # running agreement proportion at depth d
    summation = 0.0  # weighted prefix-overlap sum
    for d in range(1, max_depth + 1):
        x_d = len(set(s[:d]) & set(t[:d])) / d
        summation += (p ** (d - 1)) * x_d
    # residual: assume agreement beyond observed depth equals x_d
    rbo_ext = (1.0 - p) * summation + (p ** max_depth) * x_d
    return rbo_ext


def hit_rate_kpi(
    queries: List[str],
    retrieved_docs: List[List[Tuple[int, float]]],
    ground_truth: Dict[str, set],
    k: int = 10,
) -> Dict[str, float]:
    """Built-in KPI: document-level hit rate.

    Parameters
    ----------
    queries : List[str]
        Query strings (one per retrieval call).
    retrieved_docs : List[List[Tuple[int, float]]]
        Top-k retrieved (doc_id, score) pairs per query.
    ground_truth : Dict[str, set]
        Mapping from query string to the set of relevant doc_ids.
    k : int
        Evaluate only the top-k retrieved documents.

    Returns
    -------
    Dict[str, float]
        ``{"hit_rate@k": <value>}`` where hit_rate is the fraction of
        queries for which at least one relevant document appears in the
        top-k results.
    """
    hits = 0
    for q, docs in zip(queries, retrieved_docs):
        relevant = ground_truth.get(q, set())
        top_ids = {doc_id for doc_id, _ in docs[:k]}
        if relevant & top_ids:
            hits += 1
    return {f"hit_rate@{k}": hits / max(len(queries), 1)}


class HybridIRModelTester:
    """Main testing framework for hybrid IR models.

    Accepts either a ready-made callable *model_interface* **or** a raw
    model object together with an optional *postprocess_fn* that converts
    the model's raw output into ``List[Tuple[doc_id, score]]``.

    A user-defined KPI function can be supplied at init or at test-run
    time to compute custom performance metrics (e.g. hit-rate) using
    ground-truth relevance labels.
    """

    def __init__(
        self,
        model_interface: Callable = None,
        *,
        model: object = None,
        query_method: str = "__call__",
        postprocess_fn: Optional[Callable] = None,
        kpi_fn: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        model_interface : Callable, optional
            A function ``query -> List[Tuple[doc_id, score]]``.
            Mutually exclusive with *model*.
        model : object, optional
            Any model object.  The method named by *query_method* will be
            called with the query string.  Combine with *postprocess_fn*
            to adapt the raw output.
        query_method : str
            Name of the method to invoke on *model* (default ``"__call__"``).
        postprocess_fn : Callable, optional
            ``raw_output -> List[Tuple[doc_id, score]]``.
            Applied after every model call to normalise raw output.
        kpi_fn : Callable, optional
            Custom KPI function.  Signature::

                kpi_fn(queries, retrieved_docs, ground_truth, **kwargs)
                    -> Dict[str, float]

            See :func:`hit_rate_kpi` for an example.
        """
        if model_interface is not None:
            self._raw_model = model_interface
        elif model is not None:
            self._raw_model = getattr(model, query_method)
        else:
            raise ValueError("Provide either model_interface or model")

        self.postprocess_fn = postprocess_fn
        self.kpi_fn = kpi_fn
        self.perturbator = CharacterPerturbator()
        self.test_results = defaultdict(list)

    # --- postprocessing & KPI setters -----------------------------------

    def set_postprocess_fn(self, fn: Callable):
        """Override / set the postprocessing function."""
        self.postprocess_fn = fn

    def add_postprocess_step(self, fn: Callable):
        """Chain an additional postprocessing step after any existing one."""
        prev = self.postprocess_fn
        if prev is None:
            self.postprocess_fn = fn
        else:
            self.postprocess_fn = lambda raw, _prev=prev, _fn=fn: _fn(_prev(raw))

    def set_kpi_fn(self, fn: Callable):
        """Override / set the KPI calculation function."""
        self.kpi_fn = fn

    # --- core methods ---------------------------------------------------

    def generate_test_cases(
        self,
        queries: List[str],
        perturbation_types: List[PerturbationType] = None,
        intensity_levels: List[str] = None,
        random_type: bool = False,
    ) -> List[TestCase]:
        """Generate test cases with various perturbations.

        Parameters
        ----------
        queries : List[str]
            Input queries to perturb.
        perturbation_types : List[PerturbationType], optional
            Which perturbation types to use (default: all four).
        intensity_levels : List[str], optional
            Intensity levels (default: ``["low", "moderate", "high"]``).
        random_type : bool
            If ``True``, each query is perturbed with **one** randomly
            selected perturbation type (per intensity level) instead of
            iterating over all types.
        """
        if perturbation_types is None:
            perturbation_types = list(PerturbationType)

        if intensity_levels is None:
            intensity_levels = ["low", "moderate", "high"]

        test_cases = []

        for query in queries:
            if random_type:
                p_type = random.choice(perturbation_types)
                types_for_query = [p_type]
            else:
                types_for_query = perturbation_types

            for p_type in types_for_query:
                for intensity in intensity_levels:
                    perturbed = self.perturbator.apply_perturbation(
                        query, p_type, intensity
                    )
                    test_cases.append(TestCase(
                        original_query=query,
                        perturbed_query=perturbed,
                        perturbation_type=p_type,
                        intensity=intensity,
                    ))

        return test_cases

    def score_query(self, query: str) -> List[Tuple[int, float]]:
        """Score a single query through the model interface.

        If *postprocess_fn* is set, the raw model output is passed
        through it before being returned.
        """
        raw = self._raw_model(query)
        if self.postprocess_fn is not None:
            return self.postprocess_fn(raw)
        return raw

    def run_test_cases(
        self,
        test_cases: List[TestCase],
        *,
        kpi_fn: Optional[Callable] = None,
        ground_truth: Optional[Dict[str, set]] = None,
        kpi_kwargs: Optional[Dict] = None,
        ranking_metric: str = "rbo",
        rbo_p: float = 0.9,
    ) -> pd.DataFrame:
        """Score original and perturbed queries, compute degradation metrics.

        Parameters
        ----------
        test_cases : List[TestCase]
            Test cases produced by :meth:`generate_test_cases`.
        kpi_fn : Callable, optional
            Per-run KPI override.  Falls back to ``self.kpi_fn``.
        ground_truth : Dict[str, set], optional
            Relevance labels passed to the KPI function.  Keys are query
            strings; values are sets of relevant doc_ids.
        kpi_kwargs : dict, optional
            Extra keyword arguments forwarded to the KPI function.
        ranking_metric : str
            Which ranking-similarity metric to compute per test case.
            ``"rbo"`` (default) – Rank-Biased Overlap; works for partial
            lists (e.g. top-3).  ``"kendall_tau"`` – full-ranking
            correlation; requires the model to return scores for *all*
            documents.  ``"both"`` – compute both.
        rbo_p : float
            RBO persistence parameter (default 0.9 ≈ top-10 emphasis).

        Returns
        -------
        pd.DataFrame
            Summary indexed by intensity.  Detailed rows in
            ``self.test_results["latest"]``.
        """
        effective_kpi = kpi_fn or self.kpi_fn
        _kpi_kwargs = kpi_kwargs or {}

        use_tau = ranking_metric in ("kendall_tau", "both")
        use_rbo = ranking_metric in ("rbo", "both")

        if use_tau:
            from scipy.stats import kendalltau as _kendalltau

        rows = []
        # Cache original query scores to avoid redundant model calls
        _orig_score_cache: Dict[str, List[Tuple[int, float]]] = {}
        # Collect per-test-case results for KPI aggregation
        all_orig_queries: List[str] = []
        all_pert_queries: List[str] = []
        all_orig_results: List[List[Tuple[int, float]]] = []
        all_pert_results: List[List[Tuple[int, float]]] = []

        for tc in test_cases:
            if tc.original_query not in _orig_score_cache:
                _orig_score_cache[tc.original_query] = self.score_query(tc.original_query)
            orig_results = _orig_score_cache[tc.original_query]
            pert_results = self.score_query(tc.perturbed_query)

            all_orig_queries.append(tc.original_query)
            all_pert_queries.append(tc.perturbed_query)
            all_orig_results.append(orig_results)
            all_pert_results.append(pert_results)

            # --- RBO: works for any partial ranked list ---
            rbo_score = None
            if use_rbo:
                orig_ids = [doc_id for doc_id, _ in orig_results]
                pert_ids = [doc_id for doc_id, _ in pert_results]
                rbo_score = _rbo(orig_ids, pert_ids, p=rbo_p)

            # --- Kendall Tau: requires full doc-score vector ---
            tau = None
            if use_tau:
                n_docs = max(
                    (idx for idx, _ in orig_results + pert_results),
                    default=-1,
                ) + 1
                if n_docs > 0:
                    vec_orig = np.zeros(n_docs)
                    for idx, score in orig_results:
                        if idx < n_docs:
                            vec_orig[idx] = score
                    vec_pert = np.zeros(n_docs)
                    for idx, score in pert_results:
                        if idx < n_docs:
                            vec_pert[idx] = score
                    tau, _ = _kendalltau(vec_orig, vec_pert)
                    if np.isnan(tau):
                        tau = 1.0
                else:
                    tau = 1.0

            top1_orig = orig_results[0][1] if orig_results else 0.0
            top1_pert = pert_results[0][1] if pert_results else 0.0

            set_orig = set(idx for idx, _ in orig_results[:5])
            set_pert = set(idx for idx, _ in pert_results[:5])
            jaccard = len(set_orig & set_pert) / max(len(set_orig | set_pert), 1)

            edits = sum(1 for a, b in zip(tc.original_query, tc.perturbed_query) if a != b) + abs(
                len(tc.original_query) - len(tc.perturbed_query)
            )
            cer = edits / max(len(tc.original_query), 1)

            row = {
                "original": tc.original_query,
                "perturbed": tc.perturbed_query,
                "type": tc.perturbation_type.value,
                "intensity": tc.intensity,
                "CER": round(cer, 4),
                "top1_score_orig": round(top1_orig, 4),
                "top1_score_pert": round(top1_pert, 4),
                "top1_delta": round(top1_pert - top1_orig, 4),
                "top5_jaccard": round(jaccard, 4),
            }
            if use_tau:
                row["kendall_tau"] = round(tau, 4)
            if use_rbo:
                row["rbo"] = round(rbo_score, 4)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        self.test_results["latest"] = result_df

        # --- Aggregate performance by intensity level ---
        _ranking_cols = set()
        if use_tau:
            _ranking_cols.add("kendall_tau")
        if use_rbo:
            _ranking_cols.add("rbo")
        metric_cols = [c for c in result_df.columns
                       if c in {"CER", "top1_score_orig", "top1_score_pert",
                                "top1_delta", "top5_jaccard"} | _ranking_cols]
        summary_df = result_df.groupby("intensity")[metric_cols].mean().round(4)
        # Ensure consistent ordering: low → moderate → high
        level_order = [l for l in ["low", "moderate", "high"]
                       if l in summary_df.index]
        summary_df = summary_df.loc[level_order]

        # --- KPI evaluation per intensity level ---
        if effective_kpi is not None and ground_truth is not None:
            kpi_by_intensity = {}
            for level in level_order:
                idx = [i for i, tc in enumerate(test_cases) if tc.intensity == level]
                orig_q = [all_orig_queries[i] for i in idx]
                pert_q = [all_pert_queries[i] for i in idx]
                orig_r = [all_orig_results[i] for i in idx]
                pert_r = [all_pert_results[i] for i in idx]

                orig_kpi = effective_kpi(
                    queries=orig_q, retrieved_docs=orig_r,
                    ground_truth=ground_truth, **_kpi_kwargs,
                )
                pert_kpi = effective_kpi(
                    queries=pert_q, retrieved_docs=pert_r,
                    ground_truth=ground_truth, **_kpi_kwargs,
                )
                kpi_by_intensity[level] = {
                    "original": orig_kpi,
                    "perturbed": pert_kpi,
                }
                logger.info("KPI [%s] original:  %s", level, orig_kpi)
                logger.info("KPI [%s] perturbed: %s", level, pert_kpi)

            self.test_results["kpi"] = kpi_by_intensity

            # Append KPI deltas to summary_df
            kpi_rows = {}
            for level, kpis in kpi_by_intensity.items():
                row = {}
                for key in kpis["original"]:
                    row[f"{key}_orig"] = round(kpis["original"][key], 4)
                    row[f"{key}_pert"] = round(kpis["perturbed"][key], 4)
                    row[f"{key}_delta"] = round(
                        kpis["perturbed"][key] - kpis["original"][key], 4
                    )
                kpi_rows[level] = row
            kpi_df = pd.DataFrame(kpi_rows).T
            kpi_df.index.name = "intensity"
            summary_df = summary_df.join(kpi_df)

        self.test_results["summary"] = summary_df
        logger.info("\n%s", summary_df.to_string())

        return summary_df
    
    
