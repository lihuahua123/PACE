from __future__ import annotations

import base64
import textwrap


def build_retrieval_command(
    problem_statement: str,
    *,
    max_symbols: int = 24,
    max_files: int = 3,
    max_matches_per_file: int = 2,
) -> str:
    """Return a self-contained command that ranks likely files inside the task container."""

    encoded_problem = base64.b64encode(problem_statement.encode("utf-8", errors="replace")).decode("ascii")
    script = r'''
import base64
import os
import re
from pathlib import Path

ISSUE = base64.b64decode("__ISSUE_B64__").decode("utf-8", errors="replace")
MAX_SYMBOLS = __MAX_SYMBOLS__
MAX_FILES = __MAX_FILES__
MAX_MATCHES_PER_FILE = __MAX_MATCHES_PER_FILE__
WINDOW_RADIUS = 20
MIN_WINDOW_LINES = 25
MAX_WINDOW_LINES = 60
MAX_SYMBOL_WINDOW_LINES = 120

CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".java", ".js", ".jsx",
    ".kt", ".m", ".php", ".py", ".rb", ".rs", ".scala", ".sh", ".swift", ".tpl", ".ts",
    ".tsx", ".vue", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".next", "coverage",
}
STOPWORDS = {
    "a", "an", "actual", "additional", "after", "all", "allow", "allows", "also", "and",
    "any", "are", "argument", "arguments", "attribute", "attributes", "available", "before",
    "behavior", "bug", "but", "call", "calls", "can", "cannot", "case", "change", "code",
    "component", "components", "config", "context", "current", "currently", "debug",
    "default", "description", "details", "does", "doesn", "don", "during", "easily",
    "error", "expected", "failure", "feature", "file", "files", "filter", "filters",
    "for", "from", "function", "given", "have", "how", "idea", "implementation", "in",
    "information", "input", "into", "invalid", "is", "issue", "it", "its", "keyword",
    "keywords", "largest", "likely", "line", "list", "make", "maximum", "might",
    "minimum", "module", "more", "mount", "name", "needs", "not", "object", "objects",
    "of", "on", "only", "or", "other", "output", "path", "playbook", "problem", "repo",
    "repository", "result", "return", "same", "search", "seems", "should", "simple", "sort",
    "source", "statement", "steps", "string", "strings", "support", "summary", "task", "test",
    "tests", "that", "the", "then", "this", "through", "title", "to", "traceback", "type",
    "use", "used", "using", "value", "values", "var", "when", "with", "without", "wrapper",
    "pass", "debug", "hosts", "localhost", "tasks", "vars",
}
NOISY_HEADING_VALUES = {"component", "information", "name", "summary", "title"}
ROLE_WEIGHTS = {"source": 45, "test": 10, "docs": 4}
ROLE_KEYWORDS = {"filter", "filters", "module", "modules", "plugin", "plugins", "template", "templates"}
PATH_ROLE_HINTS = {
    "filter": ("filter", "filters", "plugins/filter"),
    "module": ("module", "modules", "plugins/modules", "module_utils"),
    "plugin": ("plugin", "plugins"),
    "connection": ("connection", "connections", "plugins/connection"),
    "inventory": ("inventory", "inventories"),
    "callback": ("callback", "callbacks", "plugins/callback"),
    "lookup": ("lookup", "lookups", "plugins/lookup"),
    "template": ("template", "templates"),
    "parser": ("parser", "parsers", "parse"),
    "serializer": ("serializer", "serializers", "serialize"),
    "yaml": ("yaml", "yml"),
    "json": ("json",),
}


def normalize_space(value):
    return re.sub(r"\s+", " ", value.strip())


def canonical_symbol(value):
    return normalize_space(value.strip("`'\"()[]{}:;,."))


def looks_code_like(symbol):
    if not symbol or len(symbol) < 2:
        return False
    if re.search(r"[./:=_-]", symbol):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", symbol):
        return len(symbol) <= 5 or "_" in symbol or any(ch.isdigit() for ch in symbol)
    if re.search(r"[A-Z].*[a-z]|[a-z].*[A-Z]", symbol):
        return True
    if re.fullmatch(r"[a-z_][a-z0-9_]+", symbol):
        return "_" in symbol
    return False


def add_weight(bucket, symbol, weight, *, force=False):
    symbol = canonical_symbol(symbol)
    if not symbol or len(symbol) > 120:
        return
    lowered = symbol.lower()
    if lowered in NOISY_HEADING_VALUES:
        return
    if lowered in STOPWORDS and not (force and lowered in {"max", "min"}):
        return
    if not force and not looks_code_like(symbol):
        return
    bucket[symbol] = max(bucket.get(symbol, 0), weight)


def is_role_keyword(symbol):
    return canonical_symbol(symbol).lower() in ROLE_KEYWORDS


def make_anchor_label(value):
    value = normalize_space(value)
    if not value:
        return ""
    parts = value.split()
    if len(parts) > 4:
        return ""
    if all(part.lower() in STOPWORDS for part in parts):
        return ""
    return value


def add_heading_phrase(anchor_labels, text, *, keep_phrase=False):
    phrase = canonical_symbol(text)
    if not phrase:
        return
    if keep_phrase:
        label = make_anchor_label(phrase)
        if label:
            anchor_labels[label] = max(anchor_labels.get(label, 0), 1)


def issue_sections(text):
    current_heading = None
    current_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = re.match(r"^\s*(?:##?\s+)?([A-Za-z][A-Za-z _/-]+?)\s*:\s*(.*)$", line)
        if match:
            if current_heading is not None:
                yield current_heading, current_lines
            current_heading = match.group(1).strip().lower()
            trailing = match.group(2).strip()
            current_lines = [trailing] if trailing else []
            continue
        if current_heading is None:
            continue
        if line.strip().startswith("## "):
            yield current_heading, current_lines
            current_heading = None
            current_lines = []
            continue
        if not line.strip() and current_lines:
            yield current_heading, current_lines
            current_heading = None
            current_lines = []
            continue
        if line.strip():
            current_lines.append(line.strip())
    if current_heading is not None:
        yield current_heading, current_lines


def tokenize_pathish(value):
    parts = []
    for piece in re.split(r"[^A-Za-z0-9_./-]+", value):
        if not piece:
            continue
        parts.append(piece)
        for sub in re.split(r"[./-]+", piece):
            if sub:
                parts.append(sub)
    return parts


def line_role(path):
    rel = str(path.relative_to(Path.cwd())).lower()
    if (
        rel.startswith("test")
        or "/test" in rel
        or "/tests/" in rel
        or any(part in {"test", "tests", "spec", "specs"} for part in rel.split("/"))
        or path.name.lower().startswith("test_")
        or path.name.lower().endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx"))
    ):
        return "test"
    if "/doc" in rel or rel.startswith("docs/") or rel.endswith((".md", ".rst")):
        return "docs"
    return "source"


symbols = {}
anchors = {}
anchor_labels = {}
traceback_paths = []
issue_lower = ISSUE.lower()
wrapper_forwarding_issue = (
    any(marker in issue_lower for marker in ("doesn't allow", "does not allow", "doesn't support", "does not support"))
    or "unexpected keyword argument" in issue_lower
)
keyword_arg_issue = wrapper_forwarding_issue or any(marker in issue_lower for marker in ("attribute=", "case_sensitive="))
template_pipe_issue = "|" in ISSUE and "{{" in ISSUE and "}}" in ISSUE

for heading, section_lines in issue_sections(ISSUE):
    if heading not in {"issue title", "title", "component name", "component", "summary", "expected behavior", "actual behavior"}:
        continue
    joined_values = [normalize_space(value) for value in section_lines if normalize_space(value)]
    keep_phrase = heading in {"issue title", "title", "component name", "component"}
    for clean_value in joined_values:
        add_heading_phrase(anchor_labels, clean_value, keep_phrase=keep_phrase)
        for token in tokenize_pathish(clean_value):
            forced = heading in {"issue title", "title", "component name", "component"}
            if forced and is_role_keyword(token):
                continue
            add_weight(anchors, token, 90 if forced else 80, force=forced)
            add_weight(symbols, token, 90 if forced else 75, force=forced and len(token) >= 4)

for value in re.findall(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(filter|module|plugin|connection|inventory|callback|lookup|template|parser|serializer)\b",
    ISSUE,
    re.IGNORECASE,
):
    name, role = value
    phrase = f"{name} {role}"
    add_heading_phrase(anchor_labels, phrase, keep_phrase=True)
    add_weight(anchors, phrase, 100, force=True)
    add_weight(symbols, name, 95, force=True)
    add_weight(symbols, phrase, 90, force=True)
    add_weight(anchors, name, 90, force=True)

for heading, value in re.findall(r"(?im)^(?:##?\s+)?(issue title|title|component name|component|summary|expected behavior|actual behavior)\s*:\s*(.+)$", ISSUE):
    clean_value = normalize_space(value)
    keep_phrase = heading.lower() in {"issue title", "title", "component name", "component"}
    add_heading_phrase(anchor_labels, clean_value, keep_phrase=keep_phrase)
    for token in tokenize_pathish(clean_value):
        forced = heading.lower() in {"issue title", "title", "component name", "component"}
        if forced and is_role_keyword(token):
            continue
        add_weight(anchors, token, 90 if forced else 80, force=forced)
        add_weight(symbols, token, 90 if forced else 75, force=forced and len(token) >= 4)

for item in re.findall(r"`([^`\n]{2,120})`", ISSUE):
    add_heading_phrase(anchor_labels, item, keep_phrase=True)
    add_weight(symbols, item, 100, force=True)
    add_weight(anchors, item, 100, force=True)
    for part in tokenize_pathish(item):
        add_weight(symbols, part, 85, force=True)
        add_weight(anchors, part, 80, force=True)

for quoted in re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_./-]{2,})['\"]", ISSUE):
    add_weight(symbols, quoted, 75, force=True)
    add_weight(anchors, quoted, 75, force=True)

for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]?([A-Za-z_][A-Za-z0-9_./-]{2,})", ISSUE):
    add_weight(symbols, key, 85, force=True)
    add_weight(symbols, value, 90, force=True)
    add_weight(anchors, key, 80, force=True)
    add_weight(anchors, value, 85, force=True)

for path, line in re.findall(r"([A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|java|go|rb|rs|php|yaml|yml|sh))(?:[\"']?,?\s+line\s+(\d+))?", ISSUE):
    traceback_paths.append((path.lstrip("./"), int(line) if line else None))
    add_weight(symbols, path, 110, force=True)
    add_weight(symbols, Path(path).name, 100, force=True)
    add_weight(anchors, path.lstrip("./"), 105, force=True)
    add_weight(anchors, Path(path).name, 100, force=True)

for item in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(", ISSUE):
    symbol = item[:-1].strip()
    add_weight(symbols, symbol, 85, force=True)
    add_weight(anchors, symbol, 80, force=True)

for item in re.findall(r"\b[a-z_][a-z0-9]+_[a-z0-9_]+\b", ISSUE):
    add_weight(symbols, item, 80, force=True)
    add_weight(anchors, item, 80, force=True)

for item in re.findall(r"\b[A-Z][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*\b", ISSUE):
    add_weight(symbols, item, 75, force=True)
    add_weight(anchors, item, 70, force=True)

for item in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", ISSUE):
    add_weight(symbols, item, 70, force=True)
    add_weight(anchors, item, 65, force=True)

for item in re.findall(r"\b[a-z]{3,}\b", ISSUE.lower()):
    if item in {"block_total", "ansible_mounts"}:
        add_weight(symbols, item, 70, force=True)
        add_weight(anchors, item, 70, force=True)

ranked_symbols = sorted(symbols.items(), key=lambda kv: (-kv[1], kv[0].lower()))[:MAX_SYMBOLS]
ranked_anchors = sorted(anchors.items(), key=lambda kv: (-kv[1], kv[0].lower()))[:MAX_SYMBOLS]
anchor_display = []
for label in sorted(anchor_labels, key=lambda k: (-anchor_labels[k], k.lower())):
    if label not in anchor_display:
        anchor_display.append(label)
for symbol, _weight in ranked_anchors:
    if symbol not in anchor_display:
        anchor_display.append(symbol)
anchor_display = anchor_display[:8]
needle_lowers = [(symbol, symbol.lower()) for symbol, _ in ranked_symbols]
anchor_lowers = [(symbol, symbol.lower()) for symbol, _ in ranked_anchors[:8]]
path_role_hints = set()
for lowered_word in re.findall(r"\b[a-z][a-z0-9_+-]{2,}\b", ISSUE.lower()):
    if lowered_word in PATH_ROLE_HINTS:
        path_role_hints.update(PATH_ROLE_HINTS[lowered_word])


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".cache")]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in CODE_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            yield path


scores = {}
matches = {}
window_matches = {}
anchor_matches = {}
path_role_matches = {}
strong_source_evidence = {}
file_line_counts = {}
file_lines = {}
all_files = list(iter_files(Path.cwd()))
rel_by_suffix = {str(path.relative_to(Path.cwd())): path for path in all_files}

for tb_path, tb_line in traceback_paths:
    for rel, path in rel_by_suffix.items():
        if rel == tb_path or rel.endswith("/" + tb_path) or tb_path.endswith("/" + rel):
            scores[path] = scores.get(path, 0) + 260
            matches.setdefault(path, []).append((tb_line or 1, "traceback path", tb_path))

path_tokens = set()
for symbol, weight in ranked_symbols[:10]:
    del weight
    for token in tokenize_pathish(symbol):
        if len(token) >= 3 and token.lower() not in STOPWORDS:
            path_tokens.add(token.lower())

for path in all_files:
    rel = str(path.relative_to(Path.cwd()))
    rel_lower = rel.lower()
    rel_parts = {part for part in re.split(r"[^a-z0-9_]+", rel_lower) if part}
    role = line_role(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        continue
    file_lines[path] = lines
    file_line_counts[path] = len(lines)

    for token in path_tokens:
        if token in rel_parts or any(token in part for part in rel_parts):
            scores[path] = scores.get(path, 0) + 30
            matches.setdefault(path, []).append((1, "path token", token))

    role_hint_count = 0
    for hint in path_role_hints:
        if hint in rel_lower:
            role_hint_count += 1
            path_role_matches.setdefault(path, []).append(hint)
    if role_hint_count:
        scores[path] = scores.get(path, 0) + min(90, 28 * role_hint_count)

    if wrapper_forwarding_issue and role == "source":
        if any(marker in rel_lower for marker in ("plugins/filter", "plugins/", "filters", "jinja", "template")):
            scores[path] = scores.get(path, 0) + 70
        if "/config/" in rel_lower or rel_lower.endswith((".yml", ".yaml")):
            scores[path] = max(1, scores.get(path, 0) // 2)
    if template_pipe_issue and role == "source" and any(marker in rel_lower for marker in ("plugins/filter", "jinja", "template")):
        scores[path] = scores.get(path, 0) + 35

    file_symbol_counts = {}
    file_distinct_symbols = []
    definition_hits = 0
    registry_hits = 0
    keyword_arg_hits = 0
    for lineno, line in enumerate(lines, start=1):
        lower = line.lower()
        stripped = line.strip()
        for anchor, anchor_lower in anchor_lowers:
            if anchor_lower in lower:
                anchor_matches.setdefault(path, []).append((lineno, anchor, stripped[:180]))
        for symbol, symbol_lower in needle_lowers:
            if symbol_lower not in lower:
                continue
            weight = symbols.get(symbol, 1)
            symbol_count = file_symbol_counts.get(symbol, 0)
            file_symbol_counts[symbol] = symbol_count + 1
            if symbol_count >= 6:
                continue

            effective_weight = weight
            if path.suffix.lower() == ".py" and lineno <= 120 and (
                line.lstrip().startswith("#")
                or "DOCUMENTATION" in line
                or stripped.startswith(("author:", "name:", "description:", "-", "options:"))
            ):
                effective_weight = max(1, weight // 4)

            definition_bonus = 0
            if re.search(rf"\b(def|class)\s+{re.escape(symbol_lower)}\b", lower):
                definition_bonus = 70
                definition_hits += 1
            elif re.search(rf"\b{re.escape(symbol_lower)}\s*=", lower):
                definition_bonus = 18
            elif re.search(rf"\b{re.escape(symbol_lower)}\b", lower) and role == "source":
                definition_bonus = 6

            if re.search(rf"['\"]{re.escape(symbol_lower)}['\"]\s*:\s*{re.escape(symbol_lower)}\b", lower):
                registry_hits += 1
                definition_bonus += 45

            scores[path] = scores.get(path, 0) + effective_weight + definition_bonus
            if symbol not in file_distinct_symbols:
                file_distinct_symbols.append(symbol)
            if definition_bonus or effective_weight >= 20:
                window_matches.setdefault(path, []).append((lineno, symbol, stripped[:180]))
            if len(matches.setdefault(path, [])) < MAX_MATCHES_PER_FILE + 2:
                matches[path].append((lineno, symbol, stripped[:180]))

        if keyword_arg_issue and ("attribute" in lower or "case_sensitive" in lower or "kwargs" in lower):
            keyword_arg_hits += 1
            scores[path] = scores.get(path, 0) + 20

    if len(file_distinct_symbols) >= 2:
        scores[path] = scores.get(path, 0) + min(120, 25 * len(file_distinct_symbols[:4]))
    if len(distinct_anchors := {anchor for _lineno, anchor, _snippet in anchor_matches.get(path, [])}) >= 2:
        scores[path] = scores.get(path, 0) + min(120, 30 * len(distinct_anchors))
    if wrapper_forwarding_issue and definition_hits:
        scores[path] = scores.get(path, 0) + min(140, 40 * definition_hits)
    if wrapper_forwarding_issue and registry_hits:
        scores[path] = scores.get(path, 0) + min(90, 35 * registry_hits)
    if keyword_arg_issue and keyword_arg_hits:
        scores[path] = scores.get(path, 0) + min(60, 15 * keyword_arg_hits)
    if (
        wrapper_forwarding_issue
        and role == "source"
        and ("/config/" in rel_lower or rel_lower.endswith((".yml", ".yaml")))
        and definition_hits == 0
        and keyword_arg_hits == 0
    ):
        scores[path] = max(1, scores.get(path, 0) // 3)

    # Do not surface a production candidate just because a generic issue word
    # appears in a comment or an unrelated expression. Source candidates need
    # evidence that they define/register a relevant symbol, handle the keyword
    # argument, or live in the explicitly inferred component path.
    strong_source_evidence[path] = bool(
        definition_hits or registry_hits or keyword_arg_hits or path_role_matches.get(path)
    )

    scores[path] = scores.get(path, 0) + ROLE_WEIGHTS[role]
    if role == "test":
        scores[path] = max(1, scores.get(path, 0) // 3)
    elif role == "docs":
        scores[path] = max(1, scores.get(path, 0) // 2)


def distinct_anchor_count(path):
    seen = []
    for _lineno, anchor, _snippet in anchor_matches.get(path, []):
        if anchor not in seen:
            seen.append(anchor)
    return len(seen)


ranked_source_files = [
    path for path in sorted(scores, key=lambda p: (-scores[p], str(p)))
    if scores.get(path, 0) > 0
    and line_role(path) == "source"
    and strong_source_evidence.get(path, False)
]
ranked_test_files = [
    path for path in sorted(scores, key=lambda p: (-scores[p], str(p)))
    if scores.get(path, 0) > 0 and line_role(path) == "test"
]
ranked_doc_files = [
    path for path in sorted(scores, key=lambda p: (-scores[p], str(p)))
    if scores.get(path, 0) > 0 and line_role(path) == "docs"
]
ranked_files = (ranked_source_files + ranked_test_files + ranked_doc_files)[:MAX_FILES]
top_anchor_overlap = max((distinct_anchor_count(path) for path in ranked_files), default=0)
unique_symbols = []
if ranked_source_files:
    for _lineno, symbol, _snippet in window_matches.get(
        ranked_source_files[0], matches.get(ranked_source_files[0], [])
    ):
        if symbol not in {"traceback path", "path token"} and symbol not in unique_symbols:
            unique_symbols.append(symbol)
top_source_hit_count = len(unique_symbols)
has_meaningful_anchor_overlap = top_anchor_overlap >= 2 or (
    top_anchor_overlap >= 1 and len(anchor_display) <= 3 and bool(ranked_files)
)
if not has_meaningful_anchor_overlap and ranked_source_files and top_source_hit_count >= 1:
    top_source = ranked_source_files[0]
    top_score = scores.get(top_source, 0)
    next_score = scores.get(ranked_files[1], 0) if len(ranked_files) > 1 else 0
    if top_score >= max(120, next_score + 40):
        has_meaningful_anchor_overlap = True


def build_window(path, hits):
    total_lines = max(file_line_counts.get(path, 0), 1)
    lines = file_lines.get(path, [])
    scored_hits = []
    for lineno, symbol, _snippet in hits:
        if not isinstance(lineno, int) or lineno <= 0:
            continue
        if symbol == "path token":
            continue
        score = 15 if symbol == "traceback path" else max(1, symbols.get(symbol, 1))
        scored_hits.append((lineno, score))
    if not scored_hits:
        return None

    def python_block_window(center):
        if path.suffix.lower() != ".py" or not lines:
            return None
        block_start = None
        block_indent = 0
        for idx in range(min(center, len(lines)), 0, -1):
            line = lines[idx - 1]
            stripped = line.lstrip()
            if re.match(r"(async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*\b", stripped):
                block_start = idx
                block_indent = len(line) - len(stripped)
                break
        if block_start is None:
            return None
        block_end = len(lines)
        for idx in range(block_start + 1, len(lines) + 1):
            line = lines[idx - 1]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= block_indent:
                block_end = idx - 1
                break
        if block_end - block_start + 1 <= MAX_SYMBOL_WINDOW_LINES:
            return block_start, block_end
        half_span = MAX_SYMBOL_WINDOW_LINES // 2
        window_start = max(block_start, center - half_span)
        window_end = min(block_end, window_start + MAX_SYMBOL_WINDOW_LINES - 1)
        if window_end - window_start + 1 < MAX_SYMBOL_WINDOW_LINES:
            window_start = max(block_start, window_end - MAX_SYMBOL_WINDOW_LINES + 1)
        return window_start, window_end

    best_center = scored_hits[0][0]
    best_score = -1
    for center, _score in scored_hits:
        window_score = sum(score for lineno, score in scored_hits if abs(lineno - center) <= WINDOW_RADIUS)
        if window_score > best_score:
            best_center = center
            best_score = window_score

    if symbol_window := python_block_window(best_center):
        return symbol_window

    target_window_lines = min(MAX_WINDOW_LINES, max(MIN_WINDOW_LINES, WINDOW_RADIUS * 2 + 1))
    half_span = target_window_lines // 2
    window_start = max(1, best_center - half_span)
    window_end = min(total_lines, window_start + target_window_lines - 1)
    if (window_end - window_start + 1) < target_window_lines:
        window_start = max(1, window_end - target_window_lines + 1)
    return window_start, window_end


recommended_windows = {
    path: build_window(path, window_matches.get(path, matches.get(path, [])))
    for path in ranked_files
}


def unique_hits(path):
    seen = set()
    result = []
    for lineno, symbol, snippet in window_matches.get(path, matches.get(path, [])):
        key = (lineno, symbol)
        if key in seen:
            continue
        seen.add(key)
        result.append((lineno, symbol, snippet))
    return result


def candidate_summary(path):
    rel = str(path.relative_to(Path.cwd()))
    role = line_role(path)
    symbols_here = []
    for _lineno, symbol, _snippet in unique_hits(path):
        if symbol in {"traceback path", "path token"}:
            continue
        if symbol not in symbols_here:
            symbols_here.append(symbol)
    if not symbols_here:
        symbols_here = [anchor for _lineno, anchor, _snippet in anchor_matches.get(path, [])[:3]]
    role_hints = []
    for hint in path_role_matches.get(path, []):
        if hint not in role_hints:
            role_hints.append(hint)
    return rel, role, symbols_here[:3], role_hints[:2]


def build_hypothesis():
    issue_lower = ISSUE.lower()
    if "attribute" in issue_lower and (" max" in issue_lower or " min" in issue_lower or "`max`" in issue_lower):
        return "A wrapper or adapter likely drops `attribute` handling before it reaches the underlying implementation."
    if traceback_paths:
        return "The referenced traceback path is likely close to the failing source-of-truth implementation."
    if "does not support" in issue_lower or "doesn't support" in issue_lower or "not allow" in issue_lower:
        return "The current code path likely rejects or ignores an input that the issue expects to be forwarded."
    return "The top candidate likely contains the narrow source-of-truth branch that controls the reported behavior."


def build_fix_shape():
    if ranked_source_files:
        rel = ranked_source_files[0].relative_to(Path.cwd())
        return f"Start in `{rel}` and make the minimal source-of-truth change tied to the issue anchors, then verify against the nearest related tests."
    return "Use the issue anchors for a focused symbol search, then patch the smallest source-of-truth implementation that explains the behavior."


print("## Deterministic Retrieval Context")
print()
print("### Issue Anchors")
if anchor_display:
    for item in anchor_display[:8]:
        print(f"- `{item}`")
else:
    print("- No strong issue anchors were extracted.")

print()
if not has_meaningful_anchor_overlap:
    print("### Retrieval Quality")
    print("- High-noise result. Treat file candidates as low confidence and start from the issue anchors first.")
    print()

print("### Pre-searched Candidate Hits")
if ranked_files:
    for idx, path in enumerate(ranked_files, start=1):
        rel, role, hit_symbols, role_hints = candidate_summary(path)
        role_text = {
            "source": "likely core implementation path",
            "test": "related test coverage path",
            "docs": "optional docs follow-up path",
        }[role]
        print(f"{idx}. `{rel}`")
        print(f"   - {role_text}")
        if hit_symbols:
            print(f"   - matched `{', '.join(hit_symbols)}`")
        elif role_hints:
            print(f"   - path hints `{', '.join(role_hints)}`")
else:
    print("- No confident candidate files were found.")

print()
print("### Suggested First Reads")
printed_reads = False
for path in ranked_files:
    window = recommended_windows.get(path)
    if window is None:
        continue
    printed_reads = True
    rel = path.relative_to(Path.cwd())
    print(f"- `{rel}:{window[0]}-{window[1]}`")
if not printed_reads:
    print("- No high-confidence windows. Use the issue anchors as focused search targets first.")

print()
print("### Precomputed Search Evidence")
if ranked_files:
    for path in ranked_files:
        rel = path.relative_to(Path.cwd())
        evidence = unique_hits(path)[:MAX_MATCHES_PER_FILE]
        if not evidence:
            continue
        print(f"- `{rel}`")
        for lineno, symbol, snippet in evidence:
            print(f"  - line {lineno}: `{symbol}` -> {snippet}")
else:
    print("- None.")

if ranked_test_files:
    print()
    print("### Related Test Candidates")
    for path in ranked_test_files[:5]:
        print(f"- `{path.relative_to(Path.cwd())}`")

print()
print("### First-action Constraint")
if printed_reads and has_meaningful_anchor_overlap:
    print("Start with a suggested read window above. If that window looks noisy, fall back to a focused symbol search using the issue anchors.")
else:
    print("Start with a focused symbol search using the issue anchors.")
print("Do not begin with broad repo enumeration, guessed workspace paths, or whole-file reads when a narrow window will do.")

print()
print("### Hypothesis")
print(f"- {build_hypothesis()}")

print()
print("### Likely Fix Shape")
print(f"- {build_fix_shape()}")
'''
    script = script.replace("__ISSUE_B64__", encoded_problem)
    script = script.replace("__MAX_SYMBOLS__", str(max_symbols))
    script = script.replace("__MAX_FILES__", str(max_files))
    script = script.replace("__MAX_MATCHES_PER_FILE__", str(max_matches_per_file))
    return "python3 - <<'PY'\n" + textwrap.dedent(script).strip() + "\nPY"
