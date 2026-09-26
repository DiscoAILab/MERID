"""GSC: derives a verified data adapter from the inspected dataset, with execution checks, recipe probes and automatic repair."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

_CODE_RE = re.compile(r"```(?:python)?\s*(.+?)```", re.DOTALL)

ADAPTER_SYSTEM = (
    "Follow the task-specific corpus schema and recipe contract supplied in the task. "
    "The MPDD layout and source names below are examples for MPDD tasks only; never impose "
    "them on another corpus. Implement a named pretrained encoder only using that actual "
    "encoder or its verified precomputed features; reject unsupported recipes instead of "
    "returning a proxy under the original encoder's name. "
    "You are an ML data engineer for a SMALL-SAMPLE (n~100-350) p>>n multimodal depression task. "
    "Write ONE Python function `load(recipe=None)` that reads the dataset into a fixed dict interface. "
    "CRITICAL — you do NOT choose the feature recipe. An OUTER loop calls load() repeatedly with different "
    "`recipe` dicts and picks the best by leak-free per-subject OOF. Implement the given recipe FAITHFULLY "
    "and DETERMINISTICALLY (same recipe -> byte-identical features every call). No 'vibes' choices, no "
    "picking extractors yourself. The `recipe` dict keys: "
    "  audio: 'wav2vec' | 'opensmile' | 'mfccs' | None -> read that Audio/<ext> dir, or DROP audio if None; "
    "  video: 'resnet' | 'densenet' | 'openface' | None -> read that Visual/<ext> dir, or DROP video if None; "
    "    (at least one of audio/video/ind_emb/meta is always present; skip a modality whose recipe value is None); "
    "  agg: per-segment frame aggregate (apply the SAME agg to audio and video), one of: "
    "       'meanstd' -> concat[mean,std]; "
    "       'meanstd_minmax' -> concat[mean,std,min,max]; "
    "       'meanstd_p25p75' -> concat[mean,std,25th percentile,75th percentile]; "
    "       'meanstd_delta' -> concat[mean,std,mean(|frame-to-frame diff|)] (captures TEMPORAL DYNAMICS; "
    "         for a 1-frame segment the delta block is zeros); "
    "  ind_emb: bool -> if True, append the per-subject individualEmbedding vector to each segment; "
    "  meta: bool -> if True, append a STRUCTURED per-subject metadata vector parsed from "
    "    personalized_train.json (train) / personalized_test.json (test) — each a dict keyed by subject id "
    "    whose entry has big5_traits, family_factors, disease. Build meta = concat[ big5_traits values in the "
    "    fixed order (Extraversion,Agreeableness,Openness,Neuroticism,Conscientiousness), "
    "    family_factors['Financial_Stress'], family_factors['Family_Members'], float(disease) ] = 8 dims. This "
    "    is the personality/context signal (the 'P' in A+V+G+P). Missing subject -> zeros; same 8-dim layout for train & test. "
    "If recipe is None, default to {'audio':'wav2vec','video':'densenet','agg':'meanstd','ind_emb':True,'meta':False} "
    "(the simplest reliable recipe — used only for the smoke test; the outer loop then calls the full grid). "
    "CASE-INSENSITIVE dir match: the extractor subdir may be cased differently than the recipe value (e.g. "
    "recipe video 'resnet' can be the directory 'ResNet'); resolve the actual subdir by listing the parent "
    "and matching the recipe name case-insensitively, so EVERY recipe value loads a non-empty array. "
    "Build train_X/test_X = concat[ audio_agg (if audio not None), video_agg (if video not None), "
    "(individualEmbedding if ind_emb), (meta 8-dim if meta) ], 2D [n,p], SAME p for train and test. Use ONLY numpy, os, glob, json. `load(recipe)` MUST return a dict with keys: "
    "train_X (float [n,p]), one int label vector per task the SPEC asks for (e.g. train_binary, "
    "and train_ternary only where that task exists), test_X (float [m,p]), "
    "test_ids (list [m]), and ALWAYS train_subject (subject id per train segment) for grouped CV. If the test "
    "set has labels, ALSO return test_binary/test_ternary aligned to test_ids. Include train_quinary/"
    "test_quinary (5-class) ONLY if a quinary/penta label exists; include train_phq/test_phq ONLY if PHQ "
    "exists. Output ONLY one ```python``` block."
)

REQUIRED = ("train_X",)


_LABEL_KEYS = ("label", "category", "phq", "score", "class", "target")


def _redact_labels(obj, redact: bool):
    if not redact:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(t in str(k).lower() for t in _LABEL_KEYS):
                out[k] = f"<{type(v).__name__} REDACTED: test label>"
            else:
                out[k] = _redact_labels(v, redact)
        return out
    if isinstance(obj, list):
        return [_redact_labels(x, redact) for x in obj[:1]]
    return obj


FEATURE_EXTS = (".npy", ".npz", ".csv", ".tsv", ".txt")


def _feature_files(d, limit=200000, skip_path=None):
    out = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if (e.is_file() and e.name.lower().endswith(FEATURE_EXTS)
                        and not (skip_path and skip_path(e.path))):
                    out.append(e.path)
                    if len(out) >= limit:
                        break
    except OSError:
        return []
    return sorted(out)


def _has_feature_file(d, skip_path=None):
    try:
        with os.scandir(d) as it:
            for e in it:
                if (e.is_file() and e.name.lower().endswith(FEATURE_EXTS)
                        and not (skip_path and skip_path(e.path))):
                    return True
    except OSError:
        pass
    return False


def _extractor_menus(root, max_depth=5, max_menus=24, skip_path=None):
    menus = {}
    for dp, dns, _fns in os.walk(root):
        if skip_path:
            dns[:] = [d for d in dns if not skip_path(os.path.join(dp, d))]
        if dp[len(root):].count(os.sep) >= max_depth:
            dns[:] = []
            continue
        exts = [d for d in sorted(dns)
                if _has_feature_file(os.path.join(dp, d), skip_path=skip_path)]
        if exts:
            menus[os.path.relpath(dp, root).replace("\\", "/")] = exts
            dns[:] = [d for d in dns if d not in exts]
        if len(menus) >= max_menus:
            break
    return menus


def _read_features(path):
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path, allow_pickle=True).astype(np.float32)
    if low.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return np.asarray(z[list(z)[0]]).astype(np.float32)
    delim = "\t" if low.endswith(".tsv") else ","
    a = np.genfromtxt(path, delimiter=delim, dtype=np.float32)
    if a.ndim == 2 and len(a) > 1 and np.isnan(a[0]).all() and not np.isnan(a[1]).all():
        a = a[1:]
    elif a.ndim == 1 and np.isnan(a).all():
        a = np.genfromtxt(path, delimiter=delim, dtype=np.float32, skip_header=1)
    return a


MEDIA_EXTS = (".wav", ".flac", ".mp3", ".mp4", ".avi", ".mov", ".mat", ".edf", ".raw",
              ".bdf", ".set", ".fif", ".cnt")


def _media_summary(root, max_dirs=12, sample=3, skip_path=None):
    out, seen, shapes, groups = [], 0, {}, {}
    for dp, dns, fns in os.walk(root):
        dns.sort()
        if skip_path:
            dns[:] = [d for d in dns if not skip_path(os.path.join(dp, d))]
            fns = [f for f in fns if not skip_path(os.path.join(dp, f))]
        med = sorted(f for f in fns if f.lower().endswith(MEDIA_EXTS))
        if not med:
            continue
        kinds = {}
        for f in med:
            kinds[os.path.splitext(f)[1].lower()] = kinds.get(os.path.splitext(f)[1].lower(), 0) + 1
        key = tuple(sorted(kinds))
        if key in shapes:
            shapes[key][0] += 1
            groups.setdefault(key, []).extend(os.path.join(dp, f) for f in med)
            continue
        seen += 1
        if seen > max_dirs:
            out.append("  MEDIA: ...(more file types not listed)")
            break
        rel = os.path.relpath(dp, root).replace(os.sep, "/")
        tot = 0
        for f in med:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
        head = "  MEDIA under %s/: %s | %.1f GB total | e.g. %s" % (
            rel, ", ".join("%d %s" % (v, k) for k, v in sorted(kinds.items())),
            tot / 1e9, ", ".join(med[:3]))
        det = []
        ext = os.path.splitext(med[0])[1].lower()
        paths = [os.path.join(dp, f) for f in med[:sample]]
        if ext in (".wav", ".flac", ".mp3"):
            try:
                import soundfile as _sf
                for q in paths:
                    i = _sf.info(q)
                    det.append("%s: %.2fs @ %dHz, %dch" % (os.path.basename(q), i.duration,
                                                           i.samplerate, i.channels))
            except ImportError:
                det.append("(soundfile not installed - duration/sample rate unknown)")
            except Exception as ex:
                det.append("(audio probe failed: %s)" % type(ex).__name__)
        elif ext == ".mat":
            try:
                from scipy.io import whosmat as _wm, loadmat as _lm
                for q in paths:
                    hdr = _wm(q)
                    det.append("%s: %s" % (os.path.basename(q), {n: sh for n, sh, _t in hdr}))
                    small = [n for n, sh, _t in hdr if int(np.prod(sh if sh else (1,))) <= 16]
                    if small:
                        try:
                            v = _lm(q, variable_names=small, squeeze_me=True)
                            det.append("      small vars: %s" % {
                                k: (v[k] if isinstance(v[k], str)
                                    else np.asarray(v[k]).ravel()[:8].tolist())
                                for k in small if k in v})
                        except Exception as ex2:
                            det.append("      (small-var read failed: %s)" % type(ex2).__name__)
            except ImportError:
                det.append("(scipy not installed - .mat contents unknown)")
            except Exception as ex:
                det.append("(mat probe failed: %s)" % type(ex).__name__)
        elif ext in (".mp4", ".avi", ".mov"):
            try:
                import cv2 as _cv
                for q in paths:
                    c = _cv.VideoCapture(q)
                    n = c.get(_cv.CAP_PROP_FRAME_COUNT); fps = c.get(_cv.CAP_PROP_FPS)
                    w = c.get(_cv.CAP_PROP_FRAME_WIDTH); h = c.get(_cv.CAP_PROP_FRAME_HEIGHT)
                    c.release()
                    det.append("%s: %.0f frames @ %.1ffps, %.0fx%.0f" % (os.path.basename(q), n, fps, w, h))
            except ImportError:
                det.append("(cv2 not installed - reporting file size only: %s)"
                           % ", ".join("%.1fMB" % (os.path.getsize(q) / 1e6) for q in paths))
            except Exception as ex:
                det.append("(video probe failed: %s)" % type(ex).__name__)
        else:
            det.append("no shape probe for %s; file sizes: %s" % (
                ext, ", ".join("%.1fMB" % (os.path.getsize(q) / 1e6) for q in paths)))
            try:
                import mne as _mne
                det.append("(mne is installed - mne.io can open %s)" % ext)
            except ImportError:
                det.append("(mne not installed - no reader for %s in this environment)" % ext)
        shapes[key] = [1, len(out)]
        groups.setdefault(key, []).extend(os.path.join(dp, f) for f in med)
        out.append(head)
        for d in det:
            out.append("    " + d)
    for cnt, at in shapes.values():
        if cnt > 1:
            out[at] += "  (+%d more directories of the same file types)" % (cnt - 1)
    try:
        import soundfile as _sf2
    except ImportError:
        _sf2 = None
    for key, (cnt, at) in shapes.items():
        if _sf2 is None or not any(k in (".wav", ".flac", ".mp3") for k in key):
            continue
        qs, bad = groups.get(key, [])[:4000], []
        for q in qs:
            try:
                _sf2.info(q)
            except Exception:
                bad.append(os.path.relpath(q, root).replace(os.sep, "/"))
        if bad:
            out[at] += ("  | %d of %d files checked FAILED to open (%s) - a reader here has to skip unreadable files rather than stop" % (len(bad), len(qs), ", ".join(bad[:4])))
        elif qs:
            out[at] += "  | %d files checked, all open" % len(qs)
    return out


def _collect_ids(id_sets, head, rows, rel):
    col = [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]
    if len(col) >= 3 and len(set(col)) >= 0.8 * len(col):
        id_sets.append((rel, str(head[0]), set(col)))


def _id_overlap_lines(id_sets):
    if len(id_sets) < 2:
        return []
    union = set().union(*(s for _r, _h, s in id_sets))
    inter = set.intersection(*(s for _r, _h, s in id_sets))
    return ["  TABLE IDS across %d tables (first column %r): %d distinct in the UNION, %d in every table; "
            "per table %s" % (len(id_sets), id_sets[0][1], len(union), len(inter),
                               ", ".join("%s=%d" % (os.path.basename(r), len(s)) for r, _h, s in id_sets))]


def _label_provenance_lines(head, rows, max_classes=10, min_rows=50, strong=0.95):
    try:
        from merid.core.grounded_state_construction.label_provenance import analyse, analyse_groups
    except Exception as exc:
        return ["    LABEL PROVENANCE: not computed (%s)" % type(exc).__name__]
    if len(rows) < min_rows:
        return []
    values, numeric = {}, []
    for j, name in enumerate(head):
        col = [r[j] if j < len(r) else "" for r in rows]
        try:
            values[name] = [float(v) for v in col]
            numeric.append(name)
        except (TypeError, ValueError):
            continue
    if len(numeric) < 3:
        return []
    out = []
    for j, name in enumerate(head):
        if name in values:
            continue
        col = [r[j] if j < len(r) else "" for r in rows]
        classes = sorted({v for v in col if v != ""})
        if not (2 <= len(classes) <= max_classes) or len(col) != len(rows):
            continue
        try:
            singles, order = analyse(values, col, numeric)
            groups, _ = analyse_groups(values, col, numeric, top_singles=3, max_size=3)
        except Exception as exc:
            out.append("    LABEL PROVENANCE %r: not computed (%s)" % (name, type(exc).__name__))
            continue
        if not singles:
            continue
        best = singles[0]
        gbest = max(groups, key=lambda g: g["sum_threshold_accuracy"]) if groups else None
        out.append("    LABEL PROVENANCE of %r (%d classes, %d sampled rows): best single column %r "
                   "thresholded at %s reproduces %.2f of it%s"
                   % (name, len(order), len(rows), best["column"],
                      ", ".join("%.4g" % c for c in best["cuts"]) or "none", best["threshold_accuracy"],
                      ("; best total %s -> %.2f" % ("+".join(gbest["columns"]), gbest["sum_threshold_accuracy"]))
                      if gbest else ""))
        hit = (best["threshold_accuracy"] >= strong or
               (gbest is not None and gbest["sum_threshold_accuracy"] >= strong))
        out.append("      -> %s" % (
            "VERIFIED TARGET-DEFINING: the label is recomputed from these columns, so a model using "
            "them recovers the label instead of predicting it" if hit else
            "no column or simple total reproduces the label; on this evidence none of them is the "
            "instrument it was derived from"))
        break
    return out


def _table_summary(root, max_files=4, max_cols=40, sample=400, skip_path=None, provenance=False):
    import csv as _csv
    sigs = {}
    id_sets = []
    out, n = [], 0
    for dp, dns, fns in os.walk(root):
        if skip_path:
            dns[:] = [d for d in dns if not skip_path(os.path.join(dp, d))]
            fns = [f for f in fns if not skip_path(os.path.join(dp, f))]
        for f in sorted(fns):
            if not f.lower().endswith((".csv", ".tsv", ".xlsx")):
                continue
            q = os.path.join(dp, f)
            try:
                if f.lower().endswith(".xlsx"):
                    try:
                        import openpyxl as _px
                    except ImportError:
                        rel = os.path.relpath(q, root).replace(os.sep, "/")
                        out.append("  TABLE %s (spreadsheet): openpyxl not installed - contents "
                                   "unknown; pip install openpyxl to read it" % rel)
                        continue
                    ws = _px.load_workbook(q, read_only=True, data_only=True).worksheets[0]
                    it = ws.iter_rows(values_only=True)
                    head = ["" if c is None else str(c) for c in (next(it, None) or [])]
                    if not head or len(head) < 2:
                        continue
                    named = sum(1 for h in head if h and not _is_number(h))
                    if named < max(2, len(head) // 2):
                        continue
                    rows = []
                    for i, r in enumerate(it):
                        if i >= sample:
                            break
                        rows.append(["" if c is None else str(c) for c in r])
                    total = ws.max_row
                    rel = os.path.relpath(q, root).replace("\\", "/")
                    n += 1
                    if n > max_files:
                        out.append("  TABLE: ...(more tables not listed)")
                        return out
                    out.append("  TABLE %s (spreadsheet): %d columns x ~%d data rows"
                               % (rel, len(head), max(0, total - 1)))
                    _collect_ids(id_sets, head, rows, rel)
                    sig = tuple(head)
                    if sig in sigs:
                        out[-1] += "  (same columns as %s - not repeated)" % sigs[sig]
                        continue
                    sigs[sig] = rel
                    out.extend(_column_lines(head, rows, max_cols))
                    if provenance:
                        out.extend(_label_provenance_lines(head, rows))
                    continue
                with open(q, newline="", encoding="utf-8", errors="replace") as fh:
                    dialect = "	" if f.lower().endswith(".tsv") else ","
                    rd = _csv.reader(fh, delimiter=dialect)
                    head = next(rd, None)
                    if not head or len(head) < 2:
                        continue
                    named = sum(1 for h in head if h and not _is_number(h))
                    if named < max(2, len(head) // 2):
                        continue
                    rows = []
                    for i, r in enumerate(rd):
                        if i >= sample:
                            break
                        rows.append(r)
                    total = len(rows) + 1
                    if len(rows) >= sample:
                        with open(q, "rb") as fb:
                            total = sum(1 for _ in fb)
            except OSError:
                continue
            n += 1
            if n > max_files:
                out.append("  TABLE: ...(more csv/tsv not listed)")
                return out
            rel = os.path.relpath(q, root).replace("\\", "/")
            out.append("  TABLE %s: %d columns x ~%d data rows" % (rel, len(head), total - 1))
            _collect_ids(id_sets, head, rows, rel)
            sig = tuple(head)
            if sig in sigs:
                out[-1] += "  (same columns as %s - not repeated)" % sigs[sig]
                continue
            sigs[sig] = rel
            out.extend(_column_lines(head, rows, max_cols))
            if provenance:
                out.extend(_label_provenance_lines(head, rows))
    out.extend(_id_overlap_lines(id_sets))
    return out


def _column_lines(head, rows, max_cols):
    out = []
    for j, name in enumerate(head[:max_cols]):
        vals = [r[j] for r in rows if j < len(r) and r[j] != ""]
        if not vals:
            out.append("      %-28s (empty)" % name)
            continue
        nums = [float(v) for v in vals if _is_number(v)]
        if len(nums) == len(vals):
            out.append("      %-28s numeric, range[%g, %g], mean %.3g"
                       % (name, min(nums), max(nums), sum(nums) / len(nums)))
        else:
            uniq = sorted(set(vals))
            out.append("      %-28s categorical, %d distinct: %s"
                       % (name, len(uniq), ", ".join(map(str, uniq[:6]))
                          + (" ..." if len(uniq) > 6 else "")))
    if len(head) > max_cols:
        out.append("      ...(%d more columns)" % (len(head) - max_cols))
    return out


def _is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _numeric_text_summary(root, max_files=3, sample=200, skip_path=None):
    out, n = [], 0
    for dp, dns, fns in os.walk(root):
        if skip_path:
            dns[:] = [d for d in dns if not skip_path(os.path.join(dp, d))]
            fns = [f for f in fns if not skip_path(os.path.join(dp, f))]
        for f in sorted(fns):
            if not f.lower().endswith((".txt", ".dat", ".asc", ".csv")):
                continue
            q = os.path.join(dp, f)
            try:
                rows, bad = [], 0
                with open(q, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if i >= sample:
                            break
                        parts = line.replace(",", " ").split()
                        if parts and all(_is_number(x) for x in parts):
                            rows.append([float(x) for x in parts])
                        elif line.strip():
                            bad += 1
                if len(rows) < 3 or bad > len(rows):
                    continue
                widths = {len(r) for r in rows}
                if len(widths) != 1:
                    continue
                size = os.path.getsize(q)
                approx = int(size / max(1, sum(len(str(x)) + 1 for x in rows[0]) + 1))
            except OSError:
                continue
            n += 1
            if n > max_files:
                out.append("  NUMERIC TEXT: ...(more not listed)")
                return out
            arr = np.asarray(rows, float)
            rel = os.path.relpath(q, root).replace("\\", "/")
            out.append("  NUMERIC TEXT %s: %d columns, ~%d rows (%.1f MB), whitespace-separated, "
                       "no header; sampled values range[%g, %g]"
                       % (rel, arr.shape[1], approx, size / 1e6, arr.min(), arr.max()))
            cw = min(arr.shape[1], 8)
            out.append("      per-column over the %d sampled rows: %s%s" % (
                len(rows),
                " | ".join("c%d mean=%.4g sd=%.4g range[%.4g, %.4g]"
                           % (j, arr[:, j].mean(), arr[:, j].std(), arr[:, j].min(), arr[:, j].max())
                           for j in range(cw)),
                "" if cw == arr.shape[1] else " ...(%d more columns)" % (arr.shape[1] - cw)))
    return out


DOC_EXTS = (".pdf", ".docx", ".md", ".rst")

_DOC_KEYS = ("sampling", "sample rate", "samplerate", " hz", "hz ", "khz", "channel", "electrode",
             "resolution", "bit", "duration", "second", "minute", "format", "column", "row",
             "subject", "participant", "record", "acquisition", "filter", "band", "reference",
             "montage", "scale", "questionnaire", "cut-off", "cutoff", "threshold", "diagnos")


def _pdf_text(path, max_pages=6):
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
    except ImportError:
        return None
    try:
        return " ".join((p.extract_text() or "") for p in PdfReader(path).pages[:max_pages])
    except Exception:
        return ""


def _docx_text(path):
    import zipfile, re as _re
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except Exception:
        return ""
    return _re.sub(r"<[^>]+>", "", _re.sub(r"</w:p>", " ", xml))


def _doc_summary(root, max_docs=8, max_lines=6, skip_path=None):
    out, seen = [], set()
    for dp, dns, fns in os.walk(root):
        if skip_path:
            dns[:] = [d for d in dns if not skip_path(os.path.join(dp, d))]
            fns = [f for f in fns if not skip_path(os.path.join(dp, f))]
        for f in sorted(fns):
            low = f.lower()
            if not (low.endswith(DOC_EXTS) or low.startswith("readme")):
                continue
            if low in seen:
                continue
            seen.add(low)
            if len(seen) > max_docs:
                out.append("  DOC: ...(more documents not listed)")
                return out
            q = os.path.join(dp, f)
            rel = os.path.relpath(q, root).replace(os.sep, "/")
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                txt = _pdf_text(q)
            elif ext == ".docx":
                txt = _docx_text(q)
            else:
                try:
                    with open(q, encoding="utf-8", errors="replace") as fh:
                        txt = fh.read(200000)
                except OSError:
                    txt = ""
            if txt is None:
                out.append("  DOC %s: (no PDF reader installed - pip install pypdf to read it)" % rel)
                continue
            flat = " ".join(txt.split())
            hits = []
            for s in flat.replace("; ", ". ").split(". "):
                sl = s.lower()
                if 20 < len(s) < 220 and any(k in sl for k in _DOC_KEYS):
                    hits.append(s.strip())
                if len(hits) >= max_lines:
                    break
            out.append("  DOC %s (%d chars read)" % (rel, len(flat)))
            for h in hits:
                out.append("      " + h)
            if not hits and flat:
                out.append("      " + flat[:200])
    return out


_HELD_OUT_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:test(?:ing)?|held[_ -]?out|holdout|final)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)


def _looks_held_out(path):
    return bool(_HELD_OUT_NAME_RE.search(str(path).replace("\\", "/")))


def _path_at_or_below(path, parent):
    try:
        return os.path.commonpath((os.path.realpath(path), parent)) == parent
    except (OSError, ValueError):
        return False


_ENV_LIBS = ("numpy", "scipy", "sklearn", "pandas", "soundfile", "librosa", "audioread", "mne",
             "openpyxl", "pypdf", "cv2", "torch", "opensmile")


def _env_summary():
    import importlib
    out = []
    for name in _ENV_LIBS:
        try:
            m = importlib.import_module(name)
            out.append("%s %s" % (name, getattr(m, "__version__", "ok")))
        except Exception as ex:
            out.append("%s MISSING (%s)" % (name, str(ex)[:60]))
    return "  ENVIRONMENT (importable here): " + " | ".join(out)


def inspect_dataset(paths: dict, max_entries=40) -> str:
    lines = [_env_summary()]
    seen_roots = {}
    canonical = {name: os.path.realpath(root) for name, root in paths.items() if root}
    training_roots = {real for name, real in canonical.items() if not _looks_held_out(name)}
    explicit_held_out_roots = {
        real for name, real in canonical.items()
        if _looks_held_out(name) and real not in training_roots
    }
    ordered_paths = sorted(paths.items(), key=lambda item: _looks_held_out(item[0]))
    for name, root in ordered_paths:
        explicit_held_out = _looks_held_out(name)
        real = os.path.realpath(root) if root else None
        if real in seen_roots:
            lines.append(f"[{name}] {root}  (same directory as [{seen_roots[real]}] above)")
            continue
        seen_roots[real] = name
        lines.append(f"[{name}] {root}")
        if not root or not os.path.exists(root):
            lines.append("  (missing)")
            continue
        n = 0
        for dp, dns, fns in os.walk(root):
            depth = dp[len(root):].count(os.sep)
            if depth > 3:
                dns[:] = []
                continue
            lines.append("  " + "  " * depth + os.path.basename(dp) + "/")
            for f in sorted(fns)[:3]:
                lines.append("  " + "  " * (depth + 1) + f)
                n += 1
            if n > max_entries:
                lines.append("  ...(truncated)")
                break
        if explicit_held_out:
            lines.append("  (held-out content not inspected; names only — path-name guard, not OS isolation)")
            continue

        def skip_held_out(q):
            return (_looks_held_out(os.path.relpath(q, root))
                    or any(_path_at_or_below(q, held) for held in explicit_held_out_roots))

        for rel, exts in sorted(_extractor_menus(root, skip_path=skip_held_out).items()):
            lines.append(f"  EXTRACTORS under {rel}/: {exts}")
            for e in exts:
                fs = _feature_files(os.path.join(root, rel, e), skip_path=skip_held_out)
                if not fs:
                    continue
                try:
                    a = _read_features(fs[0])
                    vals = np.concatenate([_read_features(f).ravel() for f in fs[:3]])
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        lines.append(f"    {e}: {len(fs)} files, no finite values read")
                        continue
                    lines.append(f"    {e}: shape={a.shape} ({len(fs)} files) | content mean={vals.mean():.3f} "
                                 f"std={vals.std():.3f} range[{vals.min():.2f},{vals.max():.2f}] zeros={(vals == 0).mean() * 100:.0f}%")
                except Exception as ex:
                    lines.append(f"    {e}: shape/content read error {ex}")
        lines.extend(_table_summary(root, skip_path=skip_held_out, provenance=not explicit_held_out))
        lines.extend(_numeric_text_summary(root, skip_path=skip_held_out))
        lines.extend(_media_summary(root, skip_path=skip_held_out))
        lines.extend(_doc_summary(root, skip_path=skip_held_out))
        npys = [q for q in glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
                if not skip_held_out(q)]
        if npys and not any("EXTRACTORS" in ln for ln in lines[-8:]):
            try:
                a = np.load(npys[0], allow_pickle=True)
                lines.append(f"  sample npy {os.path.basename(npys[0])}: shape={getattr(a, 'shape', None)}")
            except Exception as e:
                lines.append(f"  sample npy load error: {e}")
        for jf in [q for q in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
                   if not skip_held_out(q)][:4]:
            try:
                with open(jf, encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    k = list(d)[0]
                    lines.append(f"  json {os.path.basename(jf)}: dict, first entry {k} -> {d[k]}")
                elif isinstance(d, list) and d:
                    lines.append(f"  json {os.path.basename(jf)}: list, first item -> {d[0]}")
            except Exception as e:
                lines.append(f"  json {os.path.basename(jf)} error: {e}")
    return "\n".join(lines)


def _extract(text):
    m = _CODE_RE.findall(text or "")
    return m[-1].strip() if m else ""


def _build(code):
    ns = {"np": np, "numpy": np, "os": os, "glob": glob, "json": json}
    exec(compile(code, "<adapter>", "exec"), ns)
    if "load" not in ns or not callable(ns["load"]):
        raise ValueError("codegen did not define a callable load()")
    return ns["load"]


def smoke_adapter(data: dict) -> str:
    for k in REQUIRED:
        if k not in data:
            return f"missing key '{k}'"
    tx = np.asarray(data["train_X"])
    if tx.ndim != 2:
        return f"train_X must be 2D, got {tx.shape}"
    if tx.shape[0] == 0:
        return ("train_X is empty -- your file globbing likely matched 0 files; fix the path or "
                "filename pattern")
    n = tx.shape[0]
    split_sizes = {"train": n}
    for split in ("dev", "test"):
        if f"{split}_X" not in data:
            continue
        hx = np.asarray(data[f"{split}_X"])
        if hx.ndim != 2 or hx.shape[0] == 0:
            return f"{split}_X must be a non-empty 2D array, got {hx.shape}"
        if hx.shape[1] != tx.shape[1]:
            return f"train_X and {split}_X feature dims differ: {tx.shape[1]} vs {hx.shape[1]}"
        if not np.isfinite(hx).all():
            return f"{split}_X contains NaN/inf"
        split_sizes[split] = hx.shape[0]
    identities = {}
    for split, size in split_sizes.items():
        for suffix in ("subject", "ids"):
            key = f"{split}_{suffix}"
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=object)
            if values.ndim != 1 or len(values) != size:
                return f"{key} must be a 1D vector of length {size}, got {values.shape}"
            for value in values:
                if (value is None or isinstance(value, str) and not value.strip()
                        or isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                    return f"{key} contains a missing or non-finite identity"
            try:
                identities[split, suffix] = set(values.tolist())
            except TypeError:
                return f"{key} must contain scalar identities"
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for suffix in ("subject", "ids"):
            if identities.get((left, suffix), set()) & identities.get((right, suffix), set()):
                return f"{left}_{suffix} and {right}_{suffix} overlap across splits"
    metadata = {"X", "subject", "ids"}
    tasks = [k for k in data if k.startswith("train_") and k[6:] not in metadata]
    if not tasks:
        return ("no train_<task> label vector returned — the adapter must emit at least one task "
                "label (e.g. train_binary) alongside train_X")
    for split, size in split_sizes.items():
        prefix = split + "_"
        for key in data:
            if not key.startswith(prefix) or key[len(prefix):] in metadata:
                continue
            values = np.asarray(data[key])
            if values.ndim != 1 or len(values) != size:
                return f"{key} must be a 1D vector of length {size}, got {values.shape}"
            try:
                finite = np.isfinite(values).all()
            except TypeError:
                return f"{key} must contain numeric labels"
            if not finite:
                return f"{key} contains NaN/inf"
    for k in tasks:
        vals = np.unique(np.asarray(data[k]))
        if len(vals) < 2:
            return (f"{k} has a single distinct value ({vals[0]!r}) for all {n} rows — that is not a "
                    f"task. If this corpus does not define such a label, do not synthesise one: "
                    f"return only the label(s) it really has.")
    if not np.isfinite(tx).all():
        return "train_X contains NaN/inf"
    return ""


def recipe_probes(menus: dict, extras=()) -> list:
    if not menus:
        return []
    mods = sorted(menus)
    base = {m: menus[m][0] for m in mods}
    off = {e: False for e in extras}
    a = dict(base, agg="meanstd", **off)
    b = dict(base, agg="meanstd_minmax", **off)
    out = [("agg widens the vector", a, b)]
    if len(mods) > 1:
        c = dict(base, agg="meanstd", **off)
        c[mods[-1]] = None
        out.append(("dropping a modality removes its columns", a, c))

    for e in extras:
        on = dict(a, **{e: True})
        out.append((f"turning on {e!r} must add its columns", a, on))
    return out


def check_recipe_honored(load, probes) -> str:
    # Recipe probe: recipes that differ in a width-changing option must load with different
    # feature widths, confirming that every recipe option reaches the features.
    seen, vetoed = {}, 0
    for why, ra, rb in probes:
        try:
            Xa = np.asarray(load(ra)["train_X"], float)
            Xb = np.asarray(load(rb)["train_X"], float)
            pa, pb = Xa.shape[1], Xb.shape[1]
        except NotImplementedError:
            vetoed += 1
            continue
        except Exception as exc:
            return (f"load() raised on a probe recipe ({type(exc).__name__}: {str(exc)[:140]}). "
                    f"It must accept any recipe drawn from the declared schema — unless the source is "
                    f"one you have determined cannot be honoured at all, in which case raise "
                    f"NotImplementedError naming that source and saying why.")
        for rc, X in ((ra, Xa), (rb, Xb)):
            if X.shape[0] > 1 and X.shape[1] > 0 and not (np.nanstd(X, axis=0) > 0).any():
                return (f"load({rc}) returned {X.shape[0]} IDENTICAL rows — every one of its "
                        f"{X.shape[1]} columns is constant across examples. That is not a feature "
                        f"matrix, it is the same value repeated, and it is what a failed decode looks "
                        f"like when the failure is caught and replaced with zeros or a default. Find "
                        f"the source that produced it and let its error surface instead of "
                        f"substituting a placeholder: if a library cannot open these files, say so by "
                        f"raising, do not return silence. (Reading audio out of a video container "
                        f"often needs a decoder that soundfile does not provide.)")
        seen[why] = (pa, pb)
        if pa == pb:
            return (
                f"load() IGNORES part of its recipe. {ra} and {rb} both returned train_X with "
                f"p={pa}, but {why} — these cannot have equal width unless that part of the argument "
                f"is being discarded. Read `recipe` and build exactly what it asks for: a key set to "
                f"a value means INCLUDE that source, a key set to None means DROP it entirely, and "
                f"'agg' says how many statistics per feature to emit from a source that is a SERIES "
                f"(a source that is already one value per example has nothing to aggregate, and "
                f"leaving its width unchanged is correct). Do not hardcode a feature set."
            )
    if probes and not seen:
        return (f"every probe pair named a source you vetoed ({vetoed} of {len(probes)}), so nothing "
                f"could be compared and load()'s use of its recipe is UNVERIFIED. Vetoing is fine, but "
                f"the recipes that remain must still differ from one another: make sure that turning a "
                f"non-vetoed source on and off changes the columns you return. If EVERY source in the "
                f"proposed space is unusable, say so plainly by raising NotImplementedError from "
                f"load() with no recipe at all, and the run will stop instead of reporting a search "
                f"that did nothing.")
    return ""


def _development_identity_error(data):
    from merid.core.coupled_pipeline_exploration.development_search import _identities
    try:
        for split in ('train', 'dev'):
            if split + '_X' not in data:
                continue
            n = len(data[split + '_X'])
            group_key = split + '_subject' if split + '_subject' in data else split + '_ids'
            if group_key not in data:
                return split + '_subject identities are required for development selection'
            groups = _identities(data[group_key], n, group_key)
            row_key = split + '_ids'
            if row_key not in data:
                if len(set(groups)) != n:
                    return row_key + ' must identify unique source records when subjects have repeated rows'
            else:
                ids = _identities(data[row_key], n, row_key)
                if len(set(ids)) != n:
                    return row_key + ' must contain unique source-record identities'
    except ValueError as exc:
        return str(exc)
    return ''


def _scoped_adapter_load(load, data_scope):
    if data_scope == "all":
        return load
    if data_scope != "development":
        raise ValueError("data_scope must be 'all' or 'development'")

    def development_load(recipe=None):
        raw = load(recipe)
        development_load.aggregate_cache_stats = getattr(load, 'aggregate_cache_stats', None)
        scoped = {key: raw[key] for key in raw
                  if str(key).startswith("train_") or str(key).startswith("dev_")}
        if "fusion_context" in raw:
            context = raw["fusion_context"]
            scoped["fusion_context"] = {
                key: context[key] for key in
                ("evidence_names", "evidence_provenance", "train", "dev")
                if key in context
            }
        return scoped

    development_load.aggregate_cache_stats = getattr(load, 'aggregate_cache_stats', None)

    return development_load


_PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}(?:\[[A-Za-z0-9,._-]+\])?(?:==[0-9][A-Za-z0-9.*+!-]{0,39})?$")
_MOD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,79}$")
_ADAPTER_CHECK_RE = re.compile(r"^\s*CHECK:\s*([a-z_]+)\s*\((.*)\)\s*$", re.I | re.M)
_ADAPTER_CHECK_RE_ML = re.compile(r"^\s*CHECK:\s*([a-z_]+)\s*\((.*)\)\s*$", re.I | re.M | re.S)
_MAX_INSTALLS = int(os.environ.get("MERID_MAX_INSTALLS", "8"))
_ADAPTER_CHECKS = int(os.environ.get("MERID_ADAPTER_CHECKS", "4"))

_ADAPTER_INVESTIGATE = (
    "\n\nBEFORE WRITING, OR AFTER A FAILURE, YOU MAY INVESTIGATE. Emit ONE line and nothing else:\n"
    "    CHECK: <tool>(<argument>)\n"
    "You will be shown the result, then you may check again or write load(). Tools:\n%s\n"
    "You have %d checks per attempt. The ENVIRONMENT line at the top of the inspection says which "
    "libraries import here. If one you need is MISSING, CHECK: install(<pypi-name>) BEFORE writing "
    "code that imports it; what you install becomes part of this adapter's recorded environment. "
    "try_import() shows the full error chain of a failing import, so you can see which library "
    "dragged in a missing dependency and choose another. Checks never reach held-out data. When "
    "ready, answer with the ```python block only.\n"
)


def _run_site(freeze_path):
    if os.environ.get("MERID_SITE"):
        return os.environ["MERID_SITE"]
    base = (os.path.dirname(os.path.abspath(freeze_path)) if freeze_path
            else os.path.join("artifacts", "adapter_runtime"))
    return os.path.join(base, "site-packages")


def _mount_site(site):
    import importlib
    import sys as _sys
    if site and os.path.isdir(site) and site not in _sys.path:
        _sys.path.append(site)
        importlib.invalidate_caches()
        return True
    return False


def _adapter_tools(paths, state, site):
    import importlib
    import subprocess
    import sys as _sys
    import traceback as _tb
    train_root = os.path.realpath(paths.get("train_root") or ".")
    held = {os.path.realpath(p) for k, p in paths.items()
            if p and _looks_held_out(k) and os.path.realpath(p) != train_root}

    def _guard(path):
        p = os.path.abspath(str(path).strip().strip("'\""))
        try:
            rel = os.path.relpath(p, train_root)
        except ValueError:
            rel = p
        if any(_path_at_or_below(p, h) for h in held) or _looks_held_out(rel):
            raise PermissionError("held-out data is not inspectable; checks read the training side only")
        return p

    def try_import(arg):
        name = arg.strip()
        if not _MOD_RE.match(name):
            raise ValueError("give a module name, e.g. librosa or scipy.io")
        try:
            m = importlib.import_module(name)
            return {"ok": True, "version": getattr(m, "__version__", "?"),
                    "file": str(getattr(m, "__file__", ""))[-90:]}
        except Exception as exc:
            chain, e = [], exc
            while e is not None and len(chain) < 5:
                chain.append("%s: %s" % (type(e).__name__, str(e)[:160]))
                e = e.__cause__ or e.__context__
            frames = _tb.extract_tb(exc.__traceback__)[-4:]
            return {"ok": False, "error_chain": chain,
                    "raised_in": ["%s:%d" % ("/".join(f.filename.replace("\\", "/").split("/")[-2:]),
                                             f.lineno) for f in frames]}

    def install(arg):
        spec = arg.strip()
        if not _PKG_RE.match(spec):
            raise ValueError("a bare PyPI name, optionally ==version; no URLs, paths or options")
        if os.environ.get("MERID_ALLOW_INSTALL", "1") == "0":
            return {"ok": False, "reason": "installs are disabled for this run (MERID_ALLOW_INSTALL=0); "
                                           "choose a library that is already importable"}
        if state["n_installs"] >= _MAX_INSTALLS:
            return {"ok": False, "reason": "install cap reached for this run (%d)" % _MAX_INSTALLS}
        state["n_installs"] += 1
        os.makedirs(site, exist_ok=True)
        cmd = [_sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
               "--no-input", "--upgrade", "--target", site, spec]
        print("    [adapter] pip install --target %s %s" % (site, spec), flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"ok": False, "spec": spec, "reason": "pip timed out after 900s"}
        tail = ((r.stdout or "") + (r.stderr or "")).strip()[-600:]
        if r.returncode != 0:
            return {"ok": False, "spec": spec, "pip": tail}
        _mount_site(site)
        importlib.invalidate_caches()
        base = re.split(r"[\[=]", spec)[0]
        try:
            from importlib import metadata as _md
            version = _md.version(base)
        except Exception:
            version = "?"
        state["installed"].append({"spec": spec, "version": version})
        _write_env(state, site)
        return {"ok": True, "spec": spec, "version": version, "site": site,
                "note": "mounted on sys.path for this run and recorded beside the frozen adapter"}

    def traceback_(_arg=""):
        return state.get("last_tb") or "no failure recorded yet in this derivation"

    def peek(arg):
        p = _guard(arg)
        if not os.path.exists(p):
            return {"exists": False, "path": p[-120:]}
        out = {"path": p[-120:], "bytes": os.path.getsize(p)}
        with open(p, "rb") as fh:
            head = fh.read(64)
        out["head_hex"] = head[:24].hex(" ")
        out["head_ascii"] = "".join(chr(b) if 32 <= b < 127 else "." for b in head[:48])
        low = p.lower()
        if low.endswith(".mat"):
            try:
                from scipy.io import whosmat
                out["mat_vars"] = {n: (list(s), t) for n, s, t in whosmat(p)}
            except Exception as ex:
                out["mat_vars"] = "unreadable: %s" % type(ex).__name__
        elif low.endswith((".wav", ".flac", ".mp3")):
            try:
                import soundfile as _sf
                i = _sf.info(p)
                out["audio"] = {"sr": i.samplerate, "channels": i.channels,
                                "seconds": round(i.duration, 2), "subtype": i.subtype}
            except Exception as ex:
                out["audio"] = "unreadable: %s: %s" % (type(ex).__name__, str(ex)[:80])
        elif low.endswith(".npy"):
            try:
                out["npy_shape"] = list(np.load(p, mmap_mode="r").shape)
            except Exception as ex:
                out["npy_shape"] = "unreadable: %s" % type(ex).__name__
        elif low.endswith((".txt", ".csv", ".tsv", ".json", ".md")):
            with open(p, encoding="utf-8", errors="replace") as fh:
                out["first_lines"] = [fh.readline().rstrip()[:160] for _ in range(3)]
        return out

    def run(arg):
        import contextlib
        import io as _io
        snippet = arg.strip()
        if len(snippet) > 1500:
            raise ValueError("keep snippets under 1500 characters")
        import builtins as _bi
        real_open = _bi.open

        def g_open(file, *a, **k):
            mode = str(a[0]) if a else str(k.get("mode", "r"))
            if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
                _guard(os.fsdecode(file))
                return real_open(file, *a, **k)
            raise PermissionError("snippets are read-only")

        class _G:
            @staticmethod
            def glob(pattern, **k):
                return [q for q in glob.glob(pattern, **k) if not _looks_held_out(q)
                        and not any(_path_at_or_below(q, h) for h in held)]
        ns = {"np": np, "numpy": np, "os": os, "glob": _G, "json": json, "open": g_open, "re": re}
        buf = _io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(snippet, "<check>", "exec"), ns)
        except Exception as exc:
            frames = [f for f in _tb.extract_tb(exc.__traceback__) if f.filename == "<check>"]
            where = (" at line %d" % frames[-1].lineno) if frames else ""
            return {"error": "%s: %s%s" % (type(exc).__name__, str(exc)[:200], where),
                    "stdout": buf.getvalue()[-800:]}
        return {"stdout": buf.getvalue()[:1500] or "(no output; print what you want to see)"}

    return {
        "try_import": ("module name -> version, or the full error chain and where it raised", try_import),
        "install": ("PyPI name[==version] -> installs into this run's site dir and mounts it", install),
        "traceback": ("() -> the complete traceback of the last failed attempt", traceback_),
        "peek": ("path -> size, first bytes, and a header parse for .mat/.wav/.npy/text", peek),
        "run": ("python snippet -> its printed output (read-only sandbox, np/os/glob/json/re)", run),
    }


def _write_env(state, site):
    if state.get("env_path"):
        with open(state["env_path"], "w", encoding="utf-8") as fh:
            json.dump({"site": site, "installed": state["installed"]}, fh, indent=1)


def _call_tool(tools, name, arg):
    fn = tools.get(name)
    if fn is None:
        return "no such tool %r; available: %s" % (name, ", ".join(sorted(tools)))
    try:
        out = fn(arg.strip())
    except Exception as exc:
        return "%s raised %s: %s" % (name, type(exc).__name__, str(exc)[:200])
    return json.dumps(out, default=str, ensure_ascii=False)[:2000]


class AdapterAgent:
    def __init__(self, proposer):
        self.proposer = proposer

    def create_adapter(self, paths: dict, spec: str, max_iters=6, grounding="", freeze_path=None,
                       probes=None, contract_checks=(), data_scope="all", adapter_cache_bytes=0, require_row_ids=False):
        site = _run_site(freeze_path)
        state = {"last_tb": None, "installed": [], "n_installs": 0,
                 "env_path": (freeze_path + ".env.json") if freeze_path else None}
        if state["env_path"] and os.path.exists(state["env_path"]):
            try:
                prev = json.load(open(state["env_path"], encoding="utf-8"))
                state["installed"] = list(prev.get("installed", []))
                if _mount_site(prev.get("site") or site):
                    print("  [adapter] mounted recorded environment: %s" % ", ".join(
                        d.get("spec", "?") for d in state["installed"]), flush=True)
            except Exception as ex:
                print("  [adapter] could not read %s: %s" % (state["env_path"], type(ex).__name__), flush=True)
        else:
            _mount_site(site)
        if data_scope not in ("all", "development"):
            raise ValueError("data_scope must be 'all' or 'development'")
        if type(adapter_cache_bytes) is not int or adapter_cache_bytes < 0:
            raise ValueError('adapter_cache_bytes must be a nonnegative integer')
        if adapter_cache_bytes and data_scope != 'development':
            raise ValueError('adapter output caching is permitted only after development projection')

        def prepare(raw_load):
            scoped = _scoped_adapter_load(raw_load, data_scope)
            if not adapter_cache_bytes:
                return scoped
            from merid.core.grounded_state_construction.adapter_cache import AdapterOutputCache
            return AdapterOutputCache(scoped, max_bytes=adapter_cache_bytes)

        def validate(data):
            err = smoke_adapter(data)
            return err or (_development_identity_error(data) if require_row_ids else '')

        rejected_code, rejected_error = None, None
        if freeze_path and os.path.exists(freeze_path):
            try:
                with open(freeze_path, encoding="utf-8") as fh:
                    code = fh.read()
                rejected_code = code
                load = prepare(_build(code))
                data = load()
                err = validate(data)
                if not err and probes:
                    err = check_recipe_honored(load, probes)
                    for _check in (contract_checks or ()) if not err else ():
                        try:
                            err = _check(load) or ""
                        except Exception as _cx:
                            err = "%s: %s" % (type(_cx).__name__, str(_cx)[:300])
                        if err:
                            break
                if not err:
                    print(f"  [adapter] reused FROZEN adapter ({freeze_path}) — train_X "
                          f"{np.asarray(data['train_X']).shape}", flush=True)
                    return load, code, data, "(frozen adapter reused — no LLM call)"
                print(f"  [adapter] frozen adapter REJECTED: {err[:160]} -> requesting repair", flush=True)
                rejected_error = err
            except Exception as exc:
                rejected_error = type(exc).__name__ + ': ' + str(exc)[:300]
                print(f"  [adapter] frozen adapter error: {type(exc).__name__}: {str(exc)[:120]} "
                      f"-> requesting repair", flush=True)
        report = inspect_dataset(paths)
        print("== AdapterAgent: dataset inspection ==\n" + report, flush=True)
        g = (f"\nGrounding — real methods/recipes for THIS regime; anchor your extractor/aggregation/fusion "
             f"choice to these and only deviate with an explicit data-driven reason from the content stats "
             f"(not vibes like 'slightly higher' or 'comparable, choose X'):\n{grounding}\n" if grounding else "")
        prompt = (f"Dataset paths: {paths}\n\nDirectory/shape/label inspection:\n{report}\n{g}\n"
                  f"Task: {spec}\n\nWrite load() now (ONLY the ```python``` block).")
        if data_scope == 'development':
            prompt += ('\nThis invocation needs train/dev only. New or repaired code must omit test '
                       'feature/label loading; do not change the published train/dev membership. '
                       'Keep all dataset directories read-only. Disk caches may use only an explicitly '
                       'configured project cache directory from XDG_CACHE_HOME; otherwise use memory. ')
        if require_row_ids:
            prompt += ('\nRepeated records require unique train_ids/dev_ids from original source '
                       'identities such as filenames or session/segment identifiers. Preserve subject '
                       'grouping; do not invent patient identities or use array positions as source IDs. ')
        if rejected_code is not None:
            prompt += ('\nRepair this existing adapter. Preserve its valid feature extraction, label '
                       'mapping and identity ordering; make the smallest changes needed for the '
                       'reported contract error and requested data scope. Explain changed assumptions '
                       'in code comments.\nObserved error: ' + str(rejected_error) +
                       '\nExisting source:\n```python\n' + rejected_code + '\n```')
        tools = _adapter_tools(paths, state, site)
        menu = chr(10).join("    %s(%s)" % (n, d) for n, (d, _f) in sorted(tools.items()))
        calls = {n: f for n, (_d, f) in tools.items()}
        messages = [{"role": "user", "content": prompt + _ADAPTER_INVESTIGATE % (menu, _ADAPTER_CHECKS)}]
        for it in range(max_iters):
            import time as _time
            _it0 = _time.monotonic()
            print(f"  [adapter it{it + 1}/{max_iters}] asking the model for load()...", flush=True)
            spent, text = 0, ""
            for _turn in range(2 * _ADAPTER_CHECKS + 2):
                try:
                    text = self.proposer.chat(ADAPTER_SYSTEM, messages, max_tokens=12000)
                except Exception as exc:
                    print(f"  [adapter it{it + 1}] LLM error: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    return None, None, None, ""
                messages.append({"role": "assistant", "content": text})
                m = (_ADAPTER_CHECK_RE.search(text or "")
                     or _ADAPTER_CHECK_RE_ML.search(text or ""))
                if _extract(text) or not m:
                    break
                if spent >= _ADAPTER_CHECKS:
                    messages.append({"role": "user", "content":
                                     "No checks remain for this attempt. Write load() now from what you know."})
                    continue
                name, arg = m.group(1).lower(), m.group(2)
                res = _call_tool(calls, name, arg)
                ok = not res.startswith("no such tool") and " raised " not in res[:60]
                spent += 1 if ok else 0
                print(f"    [adapter checks{'' if ok else '/malformed'}] {name}({arg[:50]}) -> {res[:120]}", flush=True)
                _tail = ("That call was not well formed; re-read the tool description and try again - "
                         "it did not cost a check." if not ok else
                         "Checks left this attempt: %d. Check again, or write load() now." % (_ADAPTER_CHECKS - spent))
                messages.append({"role": "user", "content":
                                 "RESULT of CHECK: %s(%s):" % (name, arg) + chr(10) + res + chr(10) + chr(10) + _tail})
            code = _extract(text)
            if not code:
                print(f"  [adapter it{it + 1}] no code block; reply head: {(text or chr(63))[:300]!r}", flush=True)
                err = ("your reply contained NO closed ```python block"
                       + (" and no CHECK this parser could read" if "check:" in (text or "").lower()
                          else " — it was almost certainly TRUNCATED mid-code")
                       + ". Re-output ONE complete and COMPACT ```python block defining load(recipe=None); "
                         "no prose, no explanation.")
            else:
                try:
                    load = prepare(_build(code))
                    data = load()
                    err = validate(data)
                    if not err and probes:
                        err = check_recipe_honored(load, probes)
                        for _check in (contract_checks or ()) if not err else ():
                            try:
                                err = _check(load) or ""
                            except Exception as _cx:
                                err = "%s: %s" % (type(_cx).__name__, str(_cx)[:300])
                            if err:
                                break
                except Exception as exc:
                    err = f"{type(exc).__name__}: {str(exc)[:300]}"
                    import traceback as _tb
                    state["last_tb"] = "".join(_tb.format_exception(exc))[-3000:]
                    frames = [f for f in _tb.extract_tb(exc.__traceback__) if f.filename == "<adapter>"]
                    if frames:
                        f = frames[-1]
                        src_lines = (code or "").split(chr(10))
                        what = src_lines[f.lineno - 1].strip() if 0 < f.lineno <= len(src_lines) else ""
                        err += f" | raised at your load() line {f.lineno} in {f.name}(): `{what[:120]}`"
            if not err:
                widths, vetoed = "", []
                if probes:
                    ws = set()
                    for _why, ra, rb in probes:
                        for r in (ra, rb):
                            try:
                                ws.add(np.asarray(load(r)["train_X"]).shape[1])
                            except NotImplementedError as exc:
                                vetoed.append(str(exc).split(".")[0][:90])
                            except Exception:
                                pass
                    if ws:
                        widths = f", recipe-sensitive p in {sorted(ws)}"
                for v in dict.fromkeys(vetoed):
                    print(f"  [adapter] VETOED a proposed source: {v}", flush=True)
                held = ", ".join(f"{k}_X {np.asarray(data[k + '_X']).shape}"
                                 for k in ("dev", "test") if k + "_X" in data)
                print(f"  [adapter it{it + 1}] OK in {_time.monotonic() - _it0:.0f}s — train_X {np.asarray(data['train_X']).shape}"
                      + (", " + held if held else " (no held-out block — this corpus publishes none)")
                      + widths, flush=True)
                if freeze_path:
                    os.makedirs(os.path.dirname(os.path.abspath(freeze_path)), exist_ok=True)
                    with open(freeze_path, "w", encoding="utf-8") as fh:
                        fh.write(code)
                    print(f"  [adapter] FROZEN -> {freeze_path} (re-runs reuse it; delete to re-derive)",
                          flush=True)
                if state["installed"]:
                    _write_env(state, site)
                    print("  [adapter] ENVIRONMENT recorded -> %s: %s" % (state["env_path"], ", ".join(
                        "%s %s" % (d["spec"], d["version"]) for d in state["installed"])), flush=True)
                return load, code, data, text
            print(f"  [adapter it{it + 1}] FAIL after {_time.monotonic() - _it0:.0f}s -> {err}", flush=True)
            messages.append({"role": "user", "content":
                             f"load() failed: {err}" + chr(10) + "You may CHECK first (traceback() gives the full stack; try_import(x) explains a failing import; install(x) adds a missing package; peek(path) shows a file header), then re-output a corrected load()."})
        print("  [adapter] no working adapter after iterations", flush=True)
        return None, None, None, ""
