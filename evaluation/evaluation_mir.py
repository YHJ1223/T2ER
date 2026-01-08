#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIR-Bench Evaluator

This script evaluates model predictions against a ground truth dataset for medical insurance rule generation.
It supports boolean expressions (AST matching) and multi-call scenarios with parameter checking.

Features:
- AST-based logical expression matching.
- Function name and parameter validation against a schema.
- Similarity-based matching for multiple calls with the same function name.
- Generates detailed per-sample logs and aggregated summary metrics.

Usage:
    python evaluation_mir.py --dataset ./data/test.jsonl --preds ./preds/model_output.jsonl --tools ./data/tools.jsonl --output_dir ./results

Arguments:
    --dataset: Path to the dataset file (JSONL) containing 'id' and 'ground_truth'.
    --preds:   Path to the prediction file (JSONL) containing 'id', 'expr', and optionally 'calls'.
    --tools:   Path to the function library schema (JSONL).
    --output_dir: Directory where the evaluation results (details and summary) will be saved.
"""

import json
import os
import re
import ast
import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from collections import Counter
from math import isfinite

# ============================
# ========== CONFIG ==========
# ============================

# ---- Policy knobs (Logic Configuration - DO NOT CHANGE for Standard Benchmark) ----
IGNORED_PARAMS: Dict[str, Set[str]] = {
    "check_duplicate_charge": {"scope"},  # scope ignored (optional)
}

UNORDERED_PARAM_NAMES: Set[str] = {
    "child_items",
    "allowed_diagnoses",
    "fieldPaths",
    "targetValues",
    "symptoms",
}

ALLOW_PRED_EXTRA_PARAMS = False  # extra params not in schema trigger error if False
EPS = 1e-6                       # numeric tolerance

# ---- Expr fallback & matching weights ----
FALLBACK_TO_CALLS_IF_EXPR_PARSE_FAIL = True  # If expr cannot be parsed, try preds["calls"]

# Similarity scoring when multiple pred-calls share the same function name
WEIGHT_TARGET_VALUES = 0.6
WEIGHT_FIELD_PATHS   = 0.3
WEIGHT_SCALAR_EQ     = 0.1  # ratio of equal scalar params among overlapping scalar keys


# ============================
# IO helpers
# ============================

def load_any(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])
    else:
        raise ValueError(f"Unsupported file format (must be .jsonl or .json): {path}")

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def collect_json_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith((".json", ".jsonl")):
                files.append(full)
        return files
    return [path]

def path_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def resolve_output_path(base_path: str, stem: str, suffix: str) -> str:
    # If base_path implies a file (ends with json/jsonl), uses its dir.
    # Otherwise treats it as a directory.
    if base_path.lower().endswith((".json", ".jsonl")):
        out_dir = os.path.dirname(base_path) or "."
    else:
        out_dir = base_path or "."
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}{suffix}")

def match_dataset_for_pred(pred_path: str, dataset_map: Dict[str, str]) -> str:
    stem = path_stem(pred_path)
    candidates = [stem]
    
    # 1. Check for suffix match
    for suf in ("_preds", "_predictions"):
        if stem.endswith(suf):
            trimmed = stem[:-len(suf)]
            if trimmed:
                candidates.append(trimmed)
    
    # 2. Check for infix match (e.g. dataset_preds_modelname)
    for marker in ("_preds", "_predictions"):
        if marker in stem:
            parts = stem.split(marker)
            if len(parts) > 1 and parts[0]:
                candidates.append(parts[0])

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for cand in unique_candidates:
        if cand in dataset_map:
            return dataset_map[cand]
            
    raise ValueError(f"Could not find matching DATASET for {pred_path}. Ensure naming convention matches (e.g., dataset_name -> dataset_name_preds).")


# ============================
# Tools schema
# ============================

def load_tools_schema(path: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping: func_name -> {'required': [...], 'properties': {k: {'type':...}} }"""
    tools = load_any(path)
    schema = {}
    for t in tools:
        name = t.get("name")
        if not name:
            continue
        p = t.get("parameters") or {}
        schema[name] = {
            "required": list(p.get("required") or []),
            "properties": dict(p.get("properties") or {})
        }
    return schema


# ============================
# Parsing expressions to AST
# ============================

Token = Tuple[str, str]  # (type, value)
CALL_HEAD_RE = re.compile(r'([A-Za-z0-9_.]+)\s*\(')

def find_balanced_rparen(text: str, lparen_idx: int) -> int:
    depth = 0
    i = lparen_idx
    in_quote = False
    quote = ""
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == quote and text[i-1] != '\\':
                in_quote = False
        else:
            if ch in ('"', "'"):
                in_quote = True
                quote = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1

def tokenize_expr(expr: str) -> List[Token]:
    s = expr.strip()
    i = 0
    out: List[Token] = []
    while i < len(s):
        if s[i].isspace():
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            out.append(("SPACE", s[i:j]))
            i = j
            continue
        if s.startswith("and", i) and (i+3==len(s) or not s[i+3].isalnum() and s[i+3] != "_"):
            out.append(("AND", "and")); i += 3; continue
        if s.startswith("or", i) and (i+2==len(s) or not s[i+2].isalnum() and s[i+2] != "_"):
            out.append(("OR", "or")); i += 2; continue
        if s.startswith("not", i) and (i+3==len(s) or not s[i+3].isalnum() and s[i+3] != "_"):
            out.append(("NOT", "not")); i += 3; continue
        if s[i] == '(':
            out.append(("LPAREN","(")); i += 1; continue
        if s[i] == ')':
            out.append(("RPAREN",")")); i += 1; continue
        m = CALL_HEAD_RE.match(s, i)
        if m:
            lparen = s.find("(", m.end()-1)
            rparen = find_balanced_rparen(s, lparen)
            if rparen != -1:
                call_text = s[i:rparen+1]
                out.append(("CALL", call_text))
                i = rparen + 1
                continue
        j = i + 1
        out.append(("OTHER", s[i:j]))
        i = j
    return out

class Node: ...

class CallNode(Node):
    def __init__(self, text: str):
        self.text = text
    def __repr__(self): return f"CALL({self.text})"

class NotNode(Node):
    def __init__(self, child: Node):
        self.child = child
    def __repr__(self): return f"NOT({self.child})"

class AndNode(Node):
    def __init__(self, children: List[Node]):
        self.children = children
    def __repr__(self): return "AND(" + ", ".join(repr(c) for c in self.children) + ")"

class OrNode(Node):
    def __init__(self, children: List[Node]):
        self.children = children
    def __repr__(self): return "OR(" + ", ".join(repr(c) for c in self.children) + ")"

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t[0] != "SPACE"]
        self.pos = 0

    def _peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, kind: str) -> Optional[Token]:
        tok = self._peek()
        if tok and tok[0] == kind:
            self.pos += 1
            return tok
        return None

    def parse(self) -> Optional[Node]:
        if not self.tokens:
            return None
        node = self.parse_or()
        if self._peek() is not None:
            return None
        return node

    def parse_or(self) -> Optional[Node]:
        left = self.parse_and()
        if left is None: return None
        children = [left]
        while self._accept("OR"):
            right = self.parse_and()
            if right is None: return None
            children.append(right)
        return left if len(children) == 1 else OrNode(children)

    def parse_and(self) -> Optional[Node]:
        left = self.parse_unary()
        if left is None: return None
        children = [left]
        while self._accept("AND"):
            right = self.parse_unary()
            if right is None: return None
            children.append(right)
        return left if len(children) == 1 else AndNode(children)

    def parse_unary(self) -> Optional[Node]:
        if self._accept("NOT"):
            child = self.parse_unary()
            if child is None: return None
            return NotNode(child)
        return self.parse_primary()

    def parse_primary(self) -> Optional[Node]:
        tok = self._peek()
        if tok is None: return None
        if tok[0] == "CALL":
            self.pos += 1
            return CallNode(tok[1])
        if self._accept("LPAREN"):
            expr = self.parse_or()
            if expr is None or not self._accept("RPAREN"):
                return None
            return expr
        return None


# ============================
# Call parsing & normalization
# ============================

def strip_outer_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

def parse_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        val = ast.literal_eval(raw)
        return val
    except Exception:
        pass
    return strip_outer_quotes(raw)

def split_args_top(args_str: str) -> List[str]:
    parts, cur = [], []
    depth = 0
    in_quote = False
    quote = ""
    i = 0
    while i < len(args_str):
        ch = args_str[i]
        if in_quote:
            cur.append(ch)
            if ch == quote and args_str[i-1] != '\\':
                in_quote = False
        else:
            if ch in ('"', "'"):
                in_quote = True; quote = ch; cur.append(ch)
            elif ch in ('(', '[', '{'):
                depth += 1; cur.append(ch)
            elif ch in (')', ']', '}'):
                depth -= 1; cur.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(cur).strip()); cur = []
            else:
                cur.append(ch)
        i += 1
    if cur: parts.append(''.join(cur).strip())
    return parts

CALL_HEAD_RE2 = re.compile(r'([A-Za-z0-9_.]+)\s*\(')

def call_text_to_struct(call_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[str]]:
    errors: List[str] = []
    m = CALL_HEAD_RE2.match(call_text.strip())
    if not m:
        return "", {}, ["not_a_call"]
    fn_name = m.group(1).replace(".", "_")
    l = call_text.find("(", m.end()-1)
    r = find_balanced_rparen(call_text, l)
    if l == -1 or r == -1:
        return "", {}, ["paren_unbalanced"]
    args_str = call_text[l+1:r].strip()
    if not args_str:
        return fn_name, {}, errors

    tool = schema.get(fn_name) or {}
    req = list(tool.get("required") or [])
    props = list((tool.get("properties") or {}).keys())
    ordered = req + [k for k in props if k not in req and k] if props else req

    parts = split_args_top(args_str)
    has_named = any("=" in p for p in parts)
    params: Dict[str, Any] = {}

    if has_named:
        for p in parts:
            if "=" not in p:
                errors.append(f"positional_arg_not_allowed:{p}")
                continue
            k, v = p.split("=", 1)
            k = k.strip(); v = v.strip()
            pv = parse_value(v)
            params[k] = pv
    else:
        if not ordered or len(parts) > len(ordered):
            errors.append("positional_arg_overflow_or_no_schema")
        n = min(len(parts), len(ordered))
        for i in range(n):
            k = ordered[i]
            pv = parse_value(parts[i])
            params[k] = pv
        if len(parts) > n:
            for j in range(n, len(parts)):
                errors.append(f"extra_positional:{parts[j]}")

    return fn_name, params, errors


# ============================
# Call comparison utilities
# ============================

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and isfinite(x)

def number_equal(a: Any, b: Any, eps: float = EPS) -> bool:
    if is_number(a) and is_number(b):
        return abs(float(a) - float(b)) <= eps
    return False

def array_equal(a: Any, b: Any, unordered: bool) -> bool:
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    if unordered:
        def norm_item(x):
            if is_number(x): return ("num", float(x))
            return ("str", str(x))
        return Counter(norm_item(x) for x in a) == Counter(norm_item(x) for x in b)
    else:
        if len(a) != len(b): return False
        for x, y in zip(a, b):
            if is_number(x) and is_number(y):
                if not number_equal(x, y): return False
            else:
                if str(x) != str(y): return False
        return True

def type_ok(val: Any, typ: str) -> bool:
    if typ == "string": return isinstance(val, str)
    if typ == "number": return is_number(val)
    if typ == "array":  return isinstance(val, list)
    if typ == "boolean":return isinstance(val, bool)
    return True

def compare_call(
    gt_fn: str,
    gt_params: Dict[str, Any],
    pred_fn: str,
    pred_params: Dict[str, Any],
    schema: Dict[str, Any],
    func_name: str
) -> Tuple[bool, Dict[str, bool], List[str]]:
    """
    Return (CallOK, submetrics, errors)
    submetrics keys: function_name_ok, req_params_ok, param_type_ok, param_value_ok, no_extra_params
    """
    errors: List[str] = []
    subs = {
        "function_name_ok": False,
        "req_params_ok": False,
        "param_type_ok": True,
        "param_value_ok": True,
        "no_extra_params": True
    }

    # 1) function name
    if gt_fn == pred_fn:
        subs["function_name_ok"] = True
    else:
        errors.append(f"fn_name_mismatch:{pred_fn}!={gt_fn}")

    # 2) required params presence (excluding ignored)
    ignored = IGNORED_PARAMS.get(func_name, set())
    tool = schema.get(func_name) or {}
    req = set(tool.get("required") or []) - ignored

    if req.issubset(set(pred_params.keys())):
        subs["req_params_ok"] = True
    else:
        missing = list(sorted(req - set(pred_params.keys())))
        errors.append(f"missing_required:{missing}")

    # 3) extra params check
    if not ALLOW_PRED_EXTRA_PARAMS:
        props = set((tool.get("properties") or {}).keys())
        extra = [k for k in pred_params.keys() if k not in props]
        if extra:
            subs["no_extra_params"] = False
            errors.append(f"extra_params:{extra}")

    # 4) type & value checks
    props = (tool.get("properties") or {})
    for k, v in pred_params.items():
        if k in ignored: 
            continue
        expected = props.get(k, {})
        typ = expected.get("type")
        if typ and not type_ok(v, typ):
            subs["param_type_ok"] = False
            errors.append(f"type_mismatch:{k} expected {typ}, got {type(v).__name__}")
            continue
        # value compare if gt provided
        if k in gt_params:
            gv = gt_params[k]
            if isinstance(v, list) or isinstance(gv, list):
                unordered = (k in UNORDERED_PARAM_NAMES)
                if not array_equal(v, gv, unordered):
                    subs["param_value_ok"] = False
                    errors.append(f"value_mismatch_list:{k}")
            elif is_number(v) and is_number(gv):
                if not number_equal(v, gv):
                    subs["param_value_ok"] = False
                    errors.append(f"value_mismatch_num:{k} {v}!={gv}")
            else:
                if str(v) != str(gv):
                    subs["param_value_ok"] = False
                    errors.append(f"value_mismatch_str:{k} {v}!={gv}")

    call_ok = all(subs.values())
    return call_ok, subs, errors

# ---------- Friendly error mapping ----------
def friendly_errors(errs: List[str]) -> List[str]:
    out = []
    for e in errs:
        if e.startswith("missing_required:"):
            out.append("Missing required parameters.")
        elif e.startswith("extra_params:"):
            out.append("Contains parameters not defined in the schema.")
        elif e.startswith("type_mismatch:"):
            # e.g. "type_mismatch:price expected number, got str"
            try:
                _, tail = e.split(":", 1)
                name = tail.split(" expected ")[0]
                exp = tail.split(" expected ")[1].split(",")[0]
                out.append(f"Type mismatch for '{name}', expected {exp}.")
            except Exception:
                out.append("Parameter type mismatch.")
        elif e.startswith("value_mismatch_num:"):
            # value_mismatch_num:price 33.0!=11.0
            try:
                _, tail = e.split(":", 1)
                name = tail.split(" ", 1)[0]
                out.append(f"The '{name}' parameter value is incorrect.")
            except Exception:
                out.append("Numeric value mismatch.")
        elif e.startswith("value_mismatch_str:"):
            try:
                _, tail = e.split(":", 1)
                name = tail.split(" ", 1)[0]
                out.append(f"The '{name}' parameter value is incorrect.")
            except Exception:
                out.append("String value mismatch.")
        elif e.startswith("value_mismatch_list:"):
            try:
                _, tail = e.split(":", 1)
                name = tail
                out.append(f"The '{name}' list parameter does not match.")
            except Exception:
                out.append("List value mismatch.")
        elif e == "no_matching_pred_call" or e == "no_matching_pred_function_name":
            out.append("No predicted call with the same function name.")
        else:
            out.append(e)
    # remove duplicates keep order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq


# ============================
# AST structure comparison
# ============================

def flatten_calls(node: Optional[Node]) -> List[str]:
    out: List[str] = []
    if not node: return out
    def rec(n: Node):
        if isinstance(n, CallNode):
            out.append(n.text)
        elif isinstance(n, NotNode):
            rec(n.child)
        elif isinstance(n, (AndNode, OrNode)):
            for c in n.children: rec(c)
    rec(node)
    return out

def parse_call_struct(call_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[str]]:
    fn, params, errs = call_text_to_struct(call_text, schema)
    return fn, params, errs

def normalize_expr_whitespace(s: str) -> str:
    out = []
    in_q = False
    q = ""
    prev_space = False
    for ch in s.strip():
        if in_q:
            out.append(ch)
            if ch == q: in_q = False
        else:
            if ch in ('"', "'"):
                in_q = True; q = ch; out.append(ch); continue
            if ch.isspace():
                if not prev_space: out.append(' ')
                prev_space = True
            else:
                prev_space = False
                out.append(ch)
    return ''.join(out)

def build_ast(expr: str) -> Optional[Node]:
    tokens = tokenize_expr(expr)
    p = Parser(tokens)
    return p.parse()

def ast_equal(gt: Node, pred: Node, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    def node_key(n: Node) -> str:
        if isinstance(n, CallNode):
            fn, _, _ = call_text_to_struct(n.text, schema)
            return f"CALL:{fn}"
        if isinstance(n, NotNode): return "NOT"
        if isinstance(n, AndNode): return "AND"
        if isinstance(n, OrNode):  return "OR"
        return "?"

    def match(gt_node: Node, pr_node: Node) -> bool:
        if type(gt_node) is not type(pr_node):
            errors.append(f"type_mismatch:{type(gt_node).__name__}!={type(pr_node).__name__}")
            return False
        if isinstance(gt_node, CallNode):
            return True
        if isinstance(gt_node, NotNode):
            return match(gt_node.child, pr_node.child)
        if isinstance(gt_node, (AndNode, OrNode)):
            g_children = list(gt_node.children)
            p_children = list(pr_node.children)
            if len(g_children) != len(p_children):
                errors.append(f"arity_mismatch:{type(gt_node).__name__} {len(g_children)}!={len(p_children)}")
                return False
            used = [False]*len(p_children)
            for gc in g_children:
                found = False
                for j, pc in enumerate(p_children):
                    if used[j]: continue
                    if node_key(gc) != node_key(pc):
                        continue
                    if match(gc, pc):
                        used[j] = True
                        found = True
                        break
                if not found:
                    errors.append(f"child_not_matched_under_{type(gt_node).__name__}")
                    return False
            return True
        return False

    ok = match(gt, pred)
    return ok, errors


# ============================
# Similarity scoring for candidates
# ============================

def jaccard(l1: Optional[List[Any]], l2: Optional[List[Any]]) -> float:
    if not isinstance(l1, list) or not isinstance(l2, list):
        return 0.0
    s1, s2 = set(map(str, l1)), set(map(str, l2))
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0

def scalar_equal_ratio(gt_params: Dict[str, Any], pred_params: Dict[str, Any]) -> float:
    # Compare non-list scalars present on both sides
    keys = set(gt_params.keys()) & set(pred_params.keys())
    scalar_keys = [k for k in keys if not isinstance(gt_params.get(k), list) and not isinstance(pred_params.get(k), list)]
    if not scalar_keys:
        return 0.0
    eq = 0
    for k in scalar_keys:
        gv, pv = gt_params[k], pred_params[k]
        if is_number(gv) and is_number(pv):
            if number_equal(gv, pv):
                eq += 1
        else:
            if str(gv) == str(pv):
                eq += 1
    return eq / len(scalar_keys)

def candidate_score(gt_params: Dict[str, Any], pred_params: Dict[str, Any]) -> float:
    s = 0.0
    s += WEIGHT_TARGET_VALUES * jaccard(gt_params.get("targetValues"), pred_params.get("targetValues"))
    s += WEIGHT_FIELD_PATHS   * jaccard(gt_params.get("fieldPaths"),   pred_params.get("fieldPaths"))
    s += WEIGHT_SCALAR_EQ     * scalar_equal_ratio(gt_params, pred_params)
    return s

def choose_best_pred_index(
    gt_fn: str,
    gt_params: Dict[str, Any],
    pred_struct_calls: List[Tuple[str,str,Dict[str,Any],List[str]]],
    used_pred: List[bool]
) -> int:
    """
    Among pred calls with the same function name and not used, pick the one with max similarity score.
    Return index in pred_struct_calls, or -1 if none.
    """
    best_idx = -1
    best_score = -1.0
    # Enumerate candidates with same function name
    for j, (p_ct, p_fn, p_params, p_errs) in enumerate(pred_struct_calls):
        if used_pred[j]: 
            continue
        if p_fn != gt_fn:
            continue
        sc = candidate_score(gt_params, p_params)
        if sc > best_score:
            best_score = sc
            best_idx = j
    return best_idx


# ============================
# Main evaluation
# ============================

def evaluate_sample(
    sid: str,
    pred_expr: str,
    pred_calls_raw: Optional[List[str]],
    gt_exprs: List[str],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    - calls_match is now ONLY based on function-name matching existence for every GT call.
    - Submetrics are always recorded (even if incorrect); unmatched function-name => all five submetrics False.
    - If pred expr fails to parse and FALLBACK_TO_CALLS_IF_EXPR_PARSE_FAIL is True, use pred_calls_raw for call-level checks.
    """
    result_best = None
    best_score_tuple = None

    pred_expr_norm = normalize_expr_whitespace(pred_expr or "").strip()
    expr_parse_failed = False

    # Build pred AST if any non-empty expr
    pred_ast = None
    if pred_expr_norm and pred_expr_norm != "[]":
        pred_ast = build_ast(pred_expr_norm)
        if pred_ast is None:
            expr_parse_failed = True

    # Pred calls texts from AST or fallback 'calls'
    if pred_ast is not None:
        pred_calls_texts = flatten_calls(pred_ast)
    else:
        if expr_parse_failed and FALLBACK_TO_CALLS_IF_EXPR_PARSE_FAIL and isinstance(pred_calls_raw, list) and pred_calls_raw:
            pred_calls_texts = list(pred_calls_raw)
        else:
            pred_calls_texts = []

    # Struct calls for pred
    pred_struct_calls = []
    for ct in pred_calls_texts:
        fn, params, errs = parse_call_struct(ct, schema)
        pred_struct_calls.append((ct, fn, params, errs))

    for idx, gt_raw in enumerate(gt_exprs):
        gt_expr_norm = normalize_expr_whitespace(gt_raw or "").strip()
        gt_ast = build_ast(gt_expr_norm) if gt_expr_norm else None
        gt_calls_texts = flatten_calls(gt_ast)

        # Parse ground-truth calls
        gt_struct_calls = []
        for ct in gt_calls_texts:
            fn, params, errs = parse_call_struct(ct, schema)
            gt_struct_calls.append((ct, fn, params, errs))

        # Expr match (structure)
        expr_match = False
        expr_errors = []
        if pred_ast and gt_ast:
            expr_match, expr_errors = ast_equal(gt_ast, pred_ast, schema)
        elif not pred_ast and not gt_ast:
            expr_match = True  # both empty
        else:
            expr_match = False
            if expr_parse_failed:
                expr_errors = ["pred_expr_parse_failed"]
                if FALLBACK_TO_CALLS_IF_EXPR_PARSE_FAIL and pred_calls_texts:
                    expr_errors.append("fallback_to_calls")
            else:
                expr_errors = ["one_empty_expr"]

        # Calls matching based on function-name presence
        calls_match = True
        call_details = []
        used_pred = [False]*len(pred_struct_calls)

        # Submetric accumulators (micro-average)
        sub_counts = Counter()
        sub_correct = Counter()

        for g_ct, g_fn, g_params, g_errs in gt_struct_calls:
            pred_j = choose_best_pred_index(g_fn, g_params, pred_struct_calls, used_pred)
            if pred_j == -1:
                # No same-function-name candidate
                calls_match = False
                # Record five submetrics all False, add friendly error
                detail = {
                    "gt_call": g_ct,
                    "pred_call": None,
                    "function_name_ok": False,
                    "req_params_ok": False,
                    "param_type_ok": False,
                    "param_value_ok": False,
                    "no_extra_params": False,
                    "errors": friendly_errors(["no_matching_pred_function_name"])
                }
                call_details.append(detail)
                # Always count submetrics
                for k, v in detail.items():
                    if k in ("function_name_ok","req_params_ok","param_type_ok","param_value_ok","no_extra_params"):
                        sub_counts[k] += 1
                        # no increment in sub_correct since False
                continue

            # Have a candidate with the same function name
            used_pred[pred_j] = True
            p_ct, p_fn, p_params, p_errs = pred_struct_calls[pred_j]

            # Evaluate submetrics via compare_call (records even if not all pass)
            _ok, subs, errs = compare_call(g_fn, g_params, p_fn, p_params, schema, g_fn)
            detail = {
                "gt_call": g_ct,
                "pred_call": p_ct,
                "function_name_ok": subs["function_name_ok"],
                "req_params_ok": subs["req_params_ok"],
                "param_type_ok": subs["param_type_ok"],
                "param_value_ok": subs["param_value_ok"],
                "no_extra_params": subs["no_extra_params"],
                "errors": friendly_errors(errs)
            }
            call_details.append(detail)

            # Count & correct per submetric (always counted)
            for k in ("function_name_ok","req_params_ok","param_type_ok","param_value_ok","no_extra_params"):
                sub_counts[k] += 1
                if detail[k]:
                    sub_correct[k] += 1

        # Determine whether all calls are perfectly matched (all five submetrics True for every call)
        perfect_calls_match = True
        for det in call_details:
            for k in ("function_name_ok", "req_params_ok", "param_type_ok", "param_value_ok", "no_extra_params"):
                if not det.get(k, False):
                    perfect_calls_match = False
                    break
            if not perfect_calls_match:
                break

        # overall now depends on expression structure match AND perfect calls match
        overall = bool(expr_match and perfect_calls_match)

        # scoring tuple: prefer overall, then expr_match, then micro sub-acc
        total_sub = sum(sub_counts.values()) if sub_counts else 0
        sub_acc = (sum(sub_correct.values())/total_sub) if total_sub else 0.0
        score_tuple = (int(overall), int(expr_match), sub_acc)

        sub_accuracies = {
            k: (sub_correct.get(k, 0) / sub_counts[k] if sub_counts.get(k, 0) > 0 else 0.0)
            for k in ("function_name_ok", "req_params_ok", "param_type_ok", "param_value_ok", "no_extra_params")
        }

        # Ensure all submetric keys are present in the output, even if count is 0
        full_sub_correct = {
            k: sub_correct.get(k, 0)
            for k in ("function_name_ok", "req_params_ok", "param_type_ok", "param_value_ok", "no_extra_params")
        }

        cur = {
            "id": sid,
            "matched_gt_index": idx,
            "overall": overall,
            "expr_match": bool(expr_match),
            "perfect_calls_match": bool(perfect_calls_match),
            "expr_pred": pred_expr_norm,
            "expr_gt": gt_expr_norm,
            "expr_errors": expr_errors,
            "calls_details": call_details,
            "submetric_totals": dict(sub_counts),
            "submetric_correct": full_sub_correct,
            "submetric_accuracy": sub_accuracies,
        }

        if (result_best is None) or (score_tuple > best_score_tuple):
            result_best = cur
            best_score_tuple = score_tuple

    # When no gt found (shouldn't happen)
    if result_best is None:
        # Ensure all submetric keys are present in the output, even if count is 0
        full_sub_correct = {
            k: 0 for k in ("function_name_ok", "req_params_ok", "param_type_ok", "param_value_ok", "no_extra_params")
        }
        result_best = {
            "id": sid,
            "matched_gt_index": None,
            "overall": False,
            "expr_match": False,
            "perfect_calls_match": False,
            "expr_pred": pred_expr_norm,
            "expr_gt": "",
            "expr_errors": ["no_ground_truth"],
            "calls_details": [],
            "submetric_totals": {},
            "submetric_correct": full_sub_correct,
            "submetric_accuracy": {},
        }
    return result_best


def aggregate_reports(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    N = len(details)
    if N == 0:
        return {
            "num_samples": 0,
            "overall_acc": 0.0,
            "expr_acc": 0.0,
            "perfect_calls_acc": 0.0,
            "submetric_macro_avg": {}
        }

    overall = sum(1 for d in details if d.get("overall")) / N
    expr_acc = sum(1 for d in details if d.get("expr_match")) / N
    perfect_calls_acc = sum(1 for d in details if d.get("perfect_calls_match")) / N

    # NEW: submetrics macro-average (mean of per-sample accuracies)
    sub_acc_sums = Counter()
    # Get all possible keys from the first sample to initialize
    submetric_keys = set(details[0].get("submetric_accuracy", {}).keys()) if details else set()

    for d in details:
        accs = d.get("submetric_accuracy") or {}
        for k in submetric_keys:
            sub_acc_sums[k] += accs.get(k, 0.0) # Add 0 if a key is missing for some reason

    sub_acc = {k: (sub_acc_sums[k] / N) for k in submetric_keys}

    return {
        "num_samples": N,
        "overall_acc": overall,
        "expr_acc": expr_acc,
        "perfect_calls_acc": perfect_calls_acc,
        "submetric_macro_avg": sub_acc
    }


# ============================
# Main (Argparse support)
# ============================

def evaluate_pair(dataset_path: str, preds_path: str, schema: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset = load_any(dataset_path)
    preds   = load_any(preds_path)

    gt_map: Dict[str, List[str]] = {}
    for row in dataset:
        sid = row.get("id")
        gts = row.get("ground_truth") or []
        if isinstance(gts, str):
            gts = [gts]
        elif isinstance(gts, list):
            gts = [str(x) for x in gts]
        else:
            gts = []
        if sid:
            gt_map[sid] = gts

    details: List[Dict[str, Any]] = []
    for pred in preds:
        sid = pred.get("id")
        if not sid:
            continue
        pred_expr  = pred.get("expr") or ""
        pred_calls = pred.get("calls") if isinstance(pred.get("calls"), list) else []
        gts = gt_map.get(sid, [])
        res = evaluate_sample(sid, pred_expr, pred_calls, gts, schema)
        details.append(res)

    summary = aggregate_reports(details)
    return details, summary

def main():
    parser = argparse.ArgumentParser(description="Med-Insurance Function-Calling Evaluator")
    parser.add_argument("--dataset", required=True, help="Path to the dataset file (or directory) with ground truth.")
    parser.add_argument("--preds", required=True, help="Path to the predictions file (or directory).")
    parser.add_argument("--tools", required=True, help="Path to the function/tool schema file.")
    parser.add_argument("--output_dir", default="eval_output", help="Directory to save evaluation results.")
    
    args = parser.parse_args()

    # Load resources
    schema  = load_tools_schema(args.tools)
    dataset_files = collect_json_files(args.dataset)
    preds_files   = collect_json_files(args.preds)

    if not preds_files:
        raise ValueError(f"No prediction files found in {args.preds}")
    if not dataset_files:
        raise ValueError(f"No dataset files found in {args.dataset}")

    # Case 1: Single file comparison (Direct file paths provided)
    if len(dataset_files) == 1 and len(preds_files) == 1 and not os.path.isdir(args.dataset) and not os.path.isdir(args.preds):
        pred_path = preds_files[0]
        dataset_path = dataset_files[0]
        
        details, summary = evaluate_pair(dataset_path, pred_path, schema)
        
        # Construct output paths based on input filename
        stem = path_stem(pred_path)
        details_out = os.path.join(args.output_dir, f"{stem}_details.jsonl")
        summary_out = os.path.join(args.output_dir, f"{stem}_summary.json")
        
        save_jsonl(details_out, details)
        save_json(summary_out, summary)
        
        print("\n===== Evaluation Summary =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nPer-sample details saved to: {details_out}")
        print(f"Summary saved to: {summary_out}")
        return

    # Case 2: Batch processing or auto-matching
    dataset_map = {path_stem(p): p for p in dataset_files}

    for pred_path in preds_files:
        # Determine which dataset to use for this prediction file
        if len(dataset_files) == 1 and os.path.isfile(dataset_files[0]) and not os.path.isdir(args.preds):
            # If explicit single dataset file but directory of preds (or single pred file), use that dataset
            ds_path = dataset_files[0]
        else:
            # Try to match based on filename convention
            ds_path = match_dataset_for_pred(pred_path, dataset_map)

        details, summary = evaluate_pair(ds_path, pred_path, schema)

        stem = path_stem(pred_path)
        details_out = resolve_output_path(args.output_dir, stem, "_details.jsonl")
        summary_out = resolve_output_path(args.output_dir, stem, "_summary.json")

        save_jsonl(details_out, details)
        save_json(summary_out, summary)

        print(f"\n===== Evaluation Summary ({os.path.basename(pred_path)}) =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nPer-sample details saved to: {details_out}")
        print(f"Summary saved to: {summary_out}")

if __name__ == "__main__":
    main()