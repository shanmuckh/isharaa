# Isharaa Text → Sign: method and implementation

Isharaa's Text → Sign mode turns English text into American Sign Language (ASL)
performed by a 3D avatar in the browser. Everything is built in this repository:
there is no external translation service, no pretrained sign-generation model and
no motion-capture data. The system has two stages:

```
English text ──► [1] NLP: English → ASL gloss ──► sign plan ──► [2] Synthesis: plan → 3D avatar animation
                  src/text2sign/translator.py      (JSON)        static/text2sign/{timeline,avatar}.js
                  src/text2sign/data/lexicon.json                src/text2sign/data/handshapes.json
```

| Component | File |
|---|---|
| Gloss translator (Python, stdlib only) | `src/text2sign/translator.py` |
| Sign lexicon: 103 signs, keyframed | `src/text2sign/data/lexicon.json` |
| Handshape inventory: 41 shapes | `src/text2sign/data/handshapes.json` |
| Evaluation (BLEU / WER) | `src/text2sign/evaluate.py`, `data/eval_set.json` |
| Avatar: body, hand model, IK | `static/text2sign/avatar.js` |
| Timeline: keyframes → motion | `static/text2sign/timeline.js` |
| UI controller | `static/text2sign/app.js` |
| HTTP API | `app.py` (`POST /api/translate`, `GET /api/sign-data`) |
| Kinematic lexicon validator | `static/text2sign/tools/validate.mjs` |
| Unit tests | `tests/test_text2sign.py` |
| Figure generators | `static/text2sign/gallery.html`, `static/text2sign/sheet.html` |

---

## 1. English → ASL gloss

ASL is not signed English. It has its own word order and drops function words.
Grammatical information that English carries in morphology, ASL often carries
lexically or on the face. The translator is a transparent, rule-based pipeline
driven by the lexicon. Every stage is recorded in a `trace` that the UI displays.

| # | Stage | What it does |
|---|---|---|
| 1 | Normalise | Unicode quotes, whitespace, sentence split; contraction expansion (`don't` → `do not`, `what's` → `what is`) |
| 2 | Tokenise | Words and numbers. Capitalised words not at the start of a sentence are flagged as proper nouns |
| 3 | Phrases | Greedy longest match of multi-word lexicon entries (`thank you` → THANK-YOU, `i love you` → I-LOVE-YOU) |
| 4 | Analyse | Sentence type (statement / WH-question / yes-no question), tense (past / future / present), negation |
| 5 | Lemmatise | Irregular table (`went` → go) plus suffix rules (`-ies`, `-es`, `-s`, `-ing`, `-ed`, consonant doubling). A candidate is accepted only if it is a lexicon entry |
| 6 | Filter | Drop articles, copula, *do*-support, tense auxiliaries, infinitival *to*, prepositions and intensifiers, each with a recorded reason |
| 7 | Reorder | Per clause: time-first, WH-final, lexical tense marking |
| 8 | Gloss | Map to sign IDs. Unknown words and names become fingerspelling `fs-WORD`; digits map to number signs |

**Reordering and tense rules**

* **Time-first.** Time signs (`cat: "time"` in the lexicon) move to the start of the clause:
  *I went to school yesterday* → `YESTERDAY ME GO SCHOOL`.
* **WH-final.** In WH-questions the WH sign moves to the end of its clause:
  *What is your name?* → `YOUR NAME WHAT`.
* **Lexical tense.** ASL has no tense inflection. If a sentence is past or future
  and no time sign already carries the tense, `FINISH` is appended (past) or
  `FUTURE` is put first. This happens once per sentence: *I ate* → `ME EAT FINISH`.
  "-ed" adjectives such as *tired* are not treated as past, because the stem test
  requires the stripped form to be a sign.
* **Clause scoping.** Commas and semicolons separate clauses, and reordering never
  crosses them: *Thank you, see you tomorrow* → `THANK-YOU TOMORROW SEE YOU`.
* **Pronouns** follow ASL indexing glosses: ME, YOU, IX-3 (pointing to a third
  person), POSS-3, WE, THEY. Object vs possessive *her* is disambiguated by the
  next word.

**Non-manual markers (NMM).** The translator emits spans over gloss indices. The
avatar realises them on the face and head:

| NMM | Span | Realisation |
|---|---|---|
| `wh` | the WH-question clause | brows furrowed |
| `yn` | the yes/no-question clause | brows raised, head forward |
| `neg` | from the negation sign to the end of its clause | head shake |

## 2. Sign representation

Signs are described **phonologically**, using the classic parameters of sign
language phonology (Stokoe 1960; Battison 1978): handshape, location,
orientation and movement. They are not stored as recorded motion.

### 2.1 Handshapes (`handshapes.json`)

Each handshape gives the flexion of each finger joint (MCP, PIP, DIP, in degrees),
finger abduction and a thumb configuration. Thumb angles are often hard to
specify directly, so a shape can instead give a **contact constraint**. For
example, O, F and flat-O have `"tt": {"at": "i.tip"}`, meaning "the thumb tip
touches the index tip".

**Contact-constrained thumb IK.** At load time a pattern-search optimiser solves
the 5 thumb degrees of freedom (CMC yaw / abduction / twist, MCP, IP). It minimises:

* the squared distance between the thumb tip and the target (minus the contact gap), plus
* a penetration penalty that keeps thumb joints out of the palm, plus
* a soft anatomical prior that keeps the thumb MCP inside the thenar eminence, plus
* a small regulariser towards the seed pose.

The JSON therefore stays declarative ("the thumb meets the index tip"), and the
geometry is computed rather than hand-tuned.

### 2.2 Signs (`lexicon.json`)

A sign is a list of keyframes per hand. Each keyframe carries:

* `hs` — handshape
* `loc` — one of 25 named body locations (forehead, chin, chest, neutral space, …) or coordinates, plus an optional `off`set
* `palm` and `fing` — palm normal and finger direction in signer-centric terms (`out`, `up`, `left+out`, …)
* `contact` — which part of the hand is placed at the location: palm, fingertips, index tip, thumb tip, back, …
* timing (`d`, `hold`, `ease`) and an optional finger `wiggle`

Keyframes are written as deltas: anything not specified carries over from the
previous frame, so a tap is just `{"off": [...]}`. The non-dominant hand can be
`"mirror"` (symmetric signs) or have its own keyframes. Each entry also has the
English words that trigger it, a grammatical category used by the translator,
and a written phonological description.

## 3. Synthesis (avatar)

* **Body and hands.** The avatar is built procedurally in three.js with no
  external 3D assets. Each hand has 15 articulated joints plus a 3-DOF thumb
  base. It wears dark clothing against light hands for contrast, following
  interpreter practice.
* **Keyframe resolution.** For each keyframe, forward kinematics on an off-screen
  "probe" hand finds where the contact point sits for that handshape and
  orientation. The wrist target is then `location − R·contact`, so the
  *fingertips* (or thumb, or palm) land on the chin, not the wrist.
* **Arm IK.** An analytic two-bone solution (shoulder–elbow–wrist) with an elbow
  pole vector that depends on hand height. In low signing space the elbow sits
  down and slightly back; at face height it drops in front of the chest, as
  human signers do, instead of winging outwards.
* **Interpolation.**
  * Wrist paths: cubic Hermite with Catmull-Rom style velocity estimates for
    *non-uniform* key times. Tangents are zeroed at holds.
  * Orientation: quaternion slerp.
  * Handshapes: per-joint angle blending.
  * Per-segment easing: smooth, linear (used for circular movements), ease-in, ease-out.
* **Co-articulation and timing.** Signs are laid end to end. The transition into
  each sign lasts `max(d, 160 ms + 1000 ms/m × distance)`, which keeps peak wrist
  speed around 1.3–1.6 m/s, the range of natural signing. An idle hand returns
  to a rest pose. Sentence boundaries insert a pause that waits until both hands
  are at rest.
* **Fingerspelling.** Letters are laid one by one at the fingerspelling location
  with a slight drift towards the dominant side. Exceptions:
  * G and H point sideways; P and Q point down.
  * J and Z have traced movements.
  * Double letters slide sideways.

  The caption highlights the current letter.

## 4. Evaluation

```bash
uv run python src/text2sign/evaluate.py --errors     # BLEU-1..4, WER, exact match, coverage
uv run python -m unittest discover tests              # 23 unit tests (rules, NMM, lexicon integrity)
node static/text2sign/tools/validate.mjs              # reach, degenerate orientation, NaN, speed checks
```

Current results on the 50-sentence seed set (`data/eval_set.json`):

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | WER | Exact | Lexicon coverage |
|---|---|---|---|---|---|---|
| 96.58 | 93.07 | 89.83 | 81.72 | 7.69 % | 86 % | 97.95 % |

**Caveats before you report these numbers.**

* The reference glosses were written to common ASL gloss conventions, but **they
  have not been validated by a Deaf or fluent ASL signer**. Get them reviewed.
* The set is small and the vocabulary overlaps the lexicon, so treat it as an
  in-domain sanity check, not a benchmark.
* For comparison with the literature, Sign Language Production work uses large
  corpora such as RWTH-PHOENIX-Weather 2014T (DGS) or How2Sign (ASL).

The remaining errors are genuine linguistic phenomena the rule set does not model:

* object topicalisation (`BOOK ME LIKE` vs `ME LIKE BOOK`)
* modal placement (`YOU CAN HELP ME`)
* compounds (`EVERY-MORNING`)
* dropped pronouns (`TOMORROW SEE YOU`)
* gloss naming (`LATER` is signed with the FUTURE sign)

**Suggested human evaluation for the paper.** Ask ASL users to watch exported
clips (the EXPORT VIDEO button) and rate *comprehensibility* and *naturalness* on
a 5-point scale, and to transcribe what they understood. Comprehension accuracy
from that transcription is the standard measure for avatar signing (Kipp et al.
2011).

## 5. Limitations

* **Vocabulary.** 103 signs; everything else is fingerspelled.
* **Grammar.** The rule-based grammar does not model topicalisation, classifier
  predicates, spatial agreement or role shift.
* **Faces and mouth.** Non-manual markers are limited to brows and head movement.
  There are no mouth morphemes and no eye-gaze shifts.
* **Sign accuracy.** Each sign was authored from a phonological description and
  approximates the citation form. Have a signer review the forms. The sign sheet
  (`/static/text2sign/sheet.html`) makes this easy.
* **Physics.** There is no collision avoidance between the hands and the body
  during transitions.

## 6. Extending the lexicon

1. Add an entry to `src/text2sign/data/lexicon.json`. The first keyframe of each
   hand needs `hs`, `loc`, `palm` and `fing`.
2. Run `node static/text2sign/tools/validate.mjs` and the unit tests.
3. Check the result on `http://localhost:8000/static/text2sign/sheet.html?signs=YOURSIGN`,
   or in the app via the dictionary or the `?sign=YOURSIGN` deep link.

Deep links for reproducible figures:

* `/?say=<text>&at=<ms>&view=front|side|close#text-to-sign` freezes the avatar mid-sentence.
* `/static/text2sign/gallery.html?view=palm|side|three` renders the handshape inventory.
