"""English -> ASL gloss translation (rule-based, lexicon-driven).

Pipeline (every stage is recorded in the result's ``trace`` so it can be shown
in the UI and reported in the paper):

  1. normalise   expand contractions, split into sentences
  2. tokenise    words / numbers; remember capitalisation (proper nouns)
  3. phrases     greedy longest-match of multi-word lexicon entries
                 ("thank you" -> THANK-YOU, "i love you" -> I-LOVE-YOU)
  4. analyse     sentence type (statement / wh-question / yes-no question),
                 tense (past / future) and negation
  5. lemmatise   inflected form -> lexicon entry (went -> GO, studies -> STUDY)
  6. filter      drop what ASL does not sign: articles, copula, auxiliaries,
                 infinitival "to", most prepositions, intensifiers
  7. reorder     ASL ordering rules: time adverbials first, WH-word last,
                 tense marked lexically (FINISH / FUTURE) when no time sign
  8. gloss       sign IDs; words with no sign are fingerspelled (fs-WORD)

Non-manual markers (eyebrows, head shake) are emitted as spans over the gloss:
  wh  -> brows furrowed over the whole WH-question
  yn  -> brows raised (and head forward) over a yes/no question
  neg -> head shake from the negation to the end of the clause
"""

import os
import re
from functools import lru_cache

from . import load_json

# ------------------------------------------------------------------ word lists
CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "shan't": "shall not",
    "ain't": "is not", "i'm": "i am", "let's": "let us", "gonna": "going to",
    "wanna": "want to", "gotta": "have to", "y'all": "you all",
}
SUFFIX_CONTRACTIONS = [("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"),
                       ("'d", " would"), ("'m", " am")]

ARTICLES = {"a", "an", "the"}
COPULA = {"am", "is", "are", "was", "were", "be", "been", "being"}
DO_AUX = {"do", "does", "did"}
MODAL_DROP = {"would", "shall", "may", "might"}
PREPOSITIONS = {"to", "of", "at", "in", "on", "for", "from", "by", "with", "about", "into",
                "as", "than", "per"}
INTENSIFIERS = {"very", "much", "so", "really", "too", "quite", "just", "also"}
FILLERS = {"um", "uh", "oh"}
CONJ_DROP = {"and", "but", "or", "that", "if", "then"}
DEMONSTRATIVE = {"this", "that", "these", "those"}

PAST_AUX = {"was", "were", "did", "had"}
FUTURE_AUX = {"will", "shall"}
NEGATORS = {"not", "no", "never"}
WH_WORDS = {"what", "where", "when", "why", "who", "whom", "how", "which"}
YN_STARTERS = DO_AUX | COPULA | {"can", "could", "will", "would", "should", "have", "has", "may", "shall"}

IRREGULAR = {
    "went": "go", "gone": "go", "ate": "eat", "eaten": "eat", "saw": "see", "seen": "see",
    "came": "come", "knew": "know", "known": "know", "thought": "think", "drank": "drink",
    "slept": "sleep", "met": "meet", "had": "have", "has": "have", "did": "do", "done": "do",
    "made": "make", "took": "take", "taken": "take", "gave": "give", "given": "give",
    "wrote": "write", "written": "write", "drove": "drive", "driven": "drive",
    "understood": "understand", "learnt": "learn", "felt": "feel", "told": "tell",
    "said": "say", "bought": "buy", "brought": "bring", "found": "find", "left": "leave",
    "got": "get", "ran": "run", "sat": "sit", "stood": "stand", "began": "begin",
    "children": "child", "people": "person", "men": "man", "women": "woman",
}
PAST_FORMS = {"went", "ate", "saw", "came", "knew", "thought", "drank", "slept", "met", "had",
              "did", "made", "took", "gave", "wrote", "drove", "understood", "learnt", "felt",
              "told", "said", "bought", "brought", "found", "left", "got", "ran", "sat",
              "stood", "began", "was", "were",
              # past participles
              "gone", "eaten", "seen", "known", "done", "taken", "given", "written", "driven",
              "been", "drunk"}


# ------------------------------------------------------------------ lexicon
@lru_cache(maxsize=4)
def _index(lex_id: int, clips_id: int):
    """word -> sign id, phrase tuple -> sign id, sign id -> category.
    Covers the hand-authored lexicon and the signs captured from real signers."""
    from . import clip_signs
    lex = {**load_json("lexicon.json")["signs"], **clip_signs()}
    words, phrases, cats = {}, {}, {}
    for sid, entry in lex.items():
        cats[sid] = entry.get("cat", "")
        for w in entry.get("words", []):
            parts = tuple(w.lower().split())
            if len(parts) == 1:
                words.setdefault(parts[0], sid)
            else:
                phrases[parts] = sid
    max_len = max((len(p) for p in phrases), default=1)
    return words, phrases, cats, max_len


def display_gloss(sid: str) -> str:
    """ASL Citizen ids carry a variant number (BREAKFAST1); hand-authored
    glosses such as IX-3 keep theirs."""
    from . import clip_signs
    return re.sub(r"\d+$", "", sid) if sid in clip_signs() else sid


def lexicon_index():
    clips = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clips.json")
    return _index(id(load_json("lexicon.json")), id(load_json("clips.json")) if os.path.exists(clips) else 0)


# ------------------------------------------------------------------ stages
def normalise(text: str) -> list[str]:
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.findall(r"[^.!?]+[.!?]*", text)
    return [s.strip() for s in sentences if re.search(r"\w", s)]


def expand_contractions(sentence: str) -> str:
    def repl(m):
        w = m.group(0)
        low = w.lower()
        if low in CONTRACTIONS:
            out = CONTRACTIONS[low]
        else:
            out = low
            for suf, rep in SUFFIX_CONTRACTIONS:
                if low.endswith(suf):
                    stem = low[: -len(suf)]
                    if suf == "n't":
                        stem = {"wo": "will", "ca": "can", "sha": "shall"}.get(stem, stem)
                    out = stem + rep
                    break
            else:
                if low.endswith("'s"):           # it's / what's / John's
                    stem = low[:-2]
                    out = stem + (" is" if stem in {"it", "what", "that", "there", "here", "he", "she", "who", "where", "how"} else "")
        return out if not w[0].isupper() else out[0].upper() + out[1:]
    return re.sub(r"[A-Za-z]+'[A-Za-z]+|\b(?:cannot|gonna|wanna|gotta)\b", repl, sentence, flags=re.I)


def tokenise(sentence: str) -> list[dict]:
    toks = []
    for i, m in enumerate(re.finditer(r"[A-Za-z]+|\d+", sentence)):
        w = m.group(0)
        toks.append({"text": w, "low": w.lower(), "proper": w[0].isupper() and i > 0 and w != "I"})
    return toks


def lemma_candidates(w: str) -> list[str]:
    cands = [w]
    if w in IRREGULAR:
        cands.append(IRREGULAR[w])
    if w.endswith("ies") and len(w) > 4:
        cands.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        cands.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        cands.append(w[:-1])
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        cands += [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            cands.append(stem[:-1])
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        cands += [stem, w[:-1]]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            cands.append(stem[:-1])
        if stem.endswith("i"):
            cands.append(stem[:-1] + "y")
    return cands


def lookup(w: str):
    """-> (sign_id or None, lemma, inflected_past)"""
    words, _, _, _ = lexicon_index()
    cands = lemma_candidates(w)
    # regular past: "-ed" form whose stem is itself a sign (liked -> like), so
    # adjectives that merely end in -ed (tired) stay present
    regular_past = w.endswith("ed") and any(c in words for c in cands[1:])
    for c in cands:
        if c in words:
            return words[c], c, w in PAST_FORMS or regular_past
    lemma = lemma_candidates(w)[1] if w in IRREGULAR else w
    return None, lemma, w in PAST_FORMS


def merge_phrases(toks: list[dict]) -> list[dict]:
    _, phrases, _, max_len = lexicon_index()
    out, i = [], 0
    while i < len(toks):
        for n in range(min(max_len, len(toks) - i), 1, -1):
            key = tuple(t["low"] for t in toks[i:i + n])
            if key in phrases:
                out.append({"text": " ".join(t["text"] for t in toks[i:i + n]), "low": " ".join(key),
                            "proper": False, "sign": phrases[key]})
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def analyse(raw: str, toks: list[dict]) -> dict:
    lows = [t["low"] for t in toks]
    question = raw.rstrip().endswith("?")
    has_wh = any(w in WH_WORDS for w in lows)
    starts_aux = bool(lows) and lows[0] in YN_STARTERS
    if has_wh and (question or lows[0] in WH_WORDS):
        stype = "wh"
    elif question or (starts_aux and len(lows) > 1 and lows[1] in {"you", "i", "we", "they", "he", "she", "it"}):
        stype = "yn"
    else:
        stype = "statement"
    tense = "present"
    if any(w in FUTURE_AUX for w in lows) or _going_to_future(lows):
        tense = "future"
    elif any(w in PAST_AUX for w in lows) or any(lookup(w)[2] for w in lows if w not in COPULA):
        tense = "past"
    return {"type": stype, "tense": tense, "negated": any(w in NEGATORS for w in lows)}


def _going_to_future(lows):
    _, _, cats, _ = lexicon_index()
    words = lexicon_index()[0]
    for i in range(len(lows) - 2):
        if lows[i] == "going" and lows[i + 1] == "to":
            sid = words.get(lows[i + 2])
            if sid and cats.get(sid) in {"verb", "modal"}:
                return True
    return False


def _filter_and_gloss(toks: list[dict], info: dict):
    """Stages 6 + 8 for one clause -> (kept gloss tokens, dropped, lemmas)."""
    _, _, cats, _ = lexicon_index()
    lows = [t["low"] for t in toks]
    kept, dropped, lemmas = [], [], []
    for i, t in enumerate(toks):
        w, nxt = t["low"], (lows[i + 1] if i + 1 < len(lows) else None)
        prev = lows[i - 1] if i else None
        reason = None
        if "sign" in t:
            kept.append({"gloss": display_gloss(t["sign"]), "sign": t["sign"], "src": t["text"], "cat": cats.get(t["sign"], "")})
            lemmas.append(t["low"])
            continue
        if w in ARTICLES:
            reason = "article"
        elif w in COPULA:
            reason = "copula"
        elif w in DO_AUX and (nxt in {"not", "you", "i", "we", "they", "he", "she", "it"} or i == 0):
            reason = "auxiliary"
        elif w in FUTURE_AUX or w in MODAL_DROP:
            reason = "tense auxiliary"
        elif w == "going" and nxt == "to" and info["tense"] == "future":
            reason = "future 'going to'"
        elif w == "to" and prev == "going" and info["tense"] == "future":
            reason = "future 'going to'"
        elif w in {"have", "has", "had"} and any(lookup(x)[2] for x in lows[i + 1:i + 4] if x not in COPULA):
            reason = "perfect auxiliary"
        elif w in PREPOSITIONS:
            reason = "preposition"
        elif w in INTENSIFIERS:
            reason = "intensifier (non-manual)"
        elif w in CONJ_DROP or w in FILLERS:
            reason = "function word"
        if reason:
            dropped.append({"word": t["text"], "reason": reason})
            continue

        # possessive/object "her": possessive when a content word follows
        if w == "her":
            sid = "POSS-3" if nxt and nxt not in NEGATORS and nxt not in COPULA else "IX-3"
            kept.append({"gloss": sid, "sign": sid, "src": t["text"], "cat": cats[sid]})
            lemmas.append("her")
            continue

        # names are fingerspelled, but places/holidays the lexicon knows are signed
        sid, lemma, _ = lookup(w)
        if t["proper"] and cats.get(sid) != "proper":
            sid, lemma = None, w
        lemmas.append(lemma)
        if sid:
            kept.append({"gloss": display_gloss(sid), "sign": sid, "src": t["text"], "cat": cats.get(sid, "")})
        elif w.isdigit():
            for d in w:
                ds = lexicon_index()[0].get(d)
                kept.append({"gloss": ds, "sign": ds, "src": d, "cat": "num"})
        elif w in DEMONSTRATIVE:
            kept.append({"gloss": "IX-3", "sign": "IX-3", "src": t["text"], "cat": "pron"})
        else:
            word = t["text"] if t["proper"] else lemma
            kept.append({"gloss": f"fs-{word.upper()}", "fs": word.upper(), "src": t["text"], "cat": "fs"})
    return kept, dropped, lemmas


def _reorder(kept: list[dict], info: dict, wh_final: bool, mark_tense: bool):
    """Stage 7 for one clause: time-first, tense marking, WH-final."""
    time_signs = [g for g in kept if g["cat"] == "time"]
    wh_signs = [g for g in kept if g["cat"] == "wh"] if wh_final else []
    moved = {id(g) for g in time_signs + wh_signs}
    core = [g for g in kept if id(g) not in moved]
    rules, tense_marker = [], []
    if time_signs:
        rules.append("time-first: " + " ".join(g["gloss"] for g in time_signs) + " moved to front")
    if mark_tense and core and not any(g["cat"] == "tense" for g in kept):
        if info["tense"] == "past":
            core.append({"gloss": "FINISH", "sign": "FINISH", "src": "(past)", "cat": "tense"})
            rules.append("past tense with no time sign: FINISH appended")
        elif info["tense"] == "future":
            tense_marker = [{"gloss": "FUTURE", "sign": "FUTURE", "src": "(future)", "cat": "tense"}]
            rules.append("future tense with no time sign: FUTURE placed first")
    if wh_signs:
        rules.append("wh-final: " + " ".join(g["gloss"] for g in wh_signs) + " moved to end")
    return tense_marker + time_signs + core + wh_signs, rules


def translate_sentence(raw: str) -> dict:
    expanded = expand_contractions(raw)
    # clauses are reordered independently ("Thank you, see you tomorrow" must
    # not front TOMORROW before THANK-YOU)
    clause_texts = [c for c in re.split(r"[,;:]", expanded) if re.search(r"\w", c)]
    clause_toks = [merge_phrases(tokenise(c)) for c in clause_texts]
    all_toks = [t for ct in clause_toks for t in ct]
    info = analyse(raw, all_toks)

    trace = {"input": raw, "expanded": expanded, "tokens": [t["text"] for t in all_toks],
             "lemmas": [], "dropped": [], "rules": [], "clauses": len(clause_toks)}
    # Tense is marked lexically once per sentence, and only when no time sign
    # (YESTERDAY, TOMORROW...) already carries it: FUTURE on the first future
    # clause, FINISH on the last past clause.
    clause_info = [analyse(clause_texts[k], toks) for k, toks in enumerate(clause_toks)]
    clause_kept = [_filter_and_gloss(toks, ci) for toks, ci in zip(clause_toks, clause_info)]
    has_time = any(g["cat"] == "time" for kept, _, _ in clause_kept for g in kept)
    past_k = [k for k, ci in enumerate(clause_info) if ci["tense"] == "past"]
    fut_k = [k for k, ci in enumerate(clause_info) if ci["tense"] == "future"]
    mark_at = set() if has_time else {past_k[-1]} if past_k else {fut_k[0]} if fut_k else set()
    gloss, nmm = [], []
    for k, toks in enumerate(clause_toks):
        cinfo = clause_info[k]
        kept, dropped, lemmas = clause_kept[k]
        is_question_clause = (
            (info["type"] == "wh" and any(g["cat"] == "wh" for g in kept))
            or (info["type"] == "yn" and k == len(clause_toks) - 1)
        )
        cg, rules = _reorder(kept, cinfo, wh_final=info["type"] == "wh" and is_question_clause, mark_tense=k in mark_at)
        trace["lemmas"] += lemmas
        trace["dropped"] += dropped
        trace["rules"] += rules
        base = len(gloss)
        gloss += cg
        if cg and is_question_clause:
            nmm.append({"type": info["type"], "from": base, "to": len(gloss) - 1})
        neg_at = next((i for i, g in enumerate(cg) if g["cat"] == "neg"), None)
        if neg_at is not None:
            nmm.append({"type": "neg", "from": base + neg_at, "to": len(gloss) - 1})
    return {
        "english": raw,
        "type": info["type"],
        "tense": info["tense"],
        "negated": info["negated"],
        "gloss": [g["gloss"] for g in gloss],
        "tokens": gloss,
        "nmm": nmm,
        "trace": trace,
    }


def translate(text: str) -> dict:
    """English text -> {sentences: [...], plan: {items, nmm}} for the avatar."""
    sentences = [translate_sentence(s) for s in normalise(text)]
    items, nmm = [], []
    for k, s in enumerate(sentences):
        if k:
            items.append({"kind": "pause", "gloss": "//"})
        base = len(items)
        for g in s["tokens"]:
            if "fs" in g:
                items.append({"kind": "fs", "word": g["fs"], "gloss": g["gloss"], "sentence": k})
            else:
                items.append({"kind": "sign", "id": g["sign"], "gloss": g["gloss"], "sentence": k})
        for span in s["nmm"]:
            nmm.append({"type": span["type"], "from": base + span["from"], "to": base + span["to"]})
    signed = sum(1 for i in items if i["kind"] == "sign")
    spelled = sum(1 for i in items if i["kind"] == "fs")
    return {
        "text": text,
        "sentences": sentences,
        "plan": {"items": items, "nmm": nmm},
        "stats": {"signs": signed, "fingerspelled": spelled,
                  "coverage": round(signed / (signed + spelled), 3) if signed + spelled else 1.0},
    }
