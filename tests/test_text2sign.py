"""Tests for the Text -> Sign engine.  Run:  uv run python -m unittest discover tests"""

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from text2sign import load_json, translate  # noqa: E402


def gloss(text):
    return " ".join(g for s in translate(text)["sentences"] for g in s["gloss"])


def sentence(text):
    return translate(text)["sentences"][0]


class GlossRules(unittest.TestCase):
    def test_wh_question_moves_wh_word_last(self):
        self.assertEqual(gloss("What is your name?"), "YOUR NAME WHAT")
        self.assertEqual(gloss("Where is the bathroom?"), "BATHROOM WHERE")

    def test_time_sign_moves_first_and_carries_tense(self):
        self.assertEqual(gloss("I went to school yesterday."), "YESTERDAY ME GO SCHOOL")
        self.assertEqual(gloss("I will help you tomorrow."), "TOMORROW ME HELP YOU")

    def test_lexical_tense_marking_without_time_sign(self):
        self.assertEqual(gloss("I ate."), "ME EAT FINISH")
        self.assertEqual(gloss("I have eaten."), "ME EAT FINISH")
        self.assertEqual(gloss("I liked the book."), "ME LIKE BOOK FINISH")
        self.assertEqual(gloss("I am going to eat."), "FUTURE ME EAT")

    def test_adjectives_ending_in_ed_are_not_past(self):
        self.assertEqual(sentence("I am tired.")["tense"], "present")
        self.assertEqual(gloss("I am tired."), "ME TIRED")

    def test_tense_marked_once_per_sentence(self):
        self.assertEqual(gloss("I went home, and I ate."), "ME GO HOME ME EAT FINISH")
        self.assertEqual(gloss("Yesterday, I was sick."), "YESTERDAY ME SICK")

    def test_reordering_does_not_cross_clauses(self):
        self.assertEqual(gloss("Thank you, see you tomorrow!"), "THANK-YOU TOMORROW SEE YOU")

    def test_function_words_dropped(self):
        s = sentence("The boy is at the school.")
        self.assertEqual(" ".join(s["gloss"]), "BOY SCHOOL")
        reasons = {d["word"].lower(): d["reason"] for d in s["trace"]["dropped"]}
        self.assertEqual(reasons["the"], "article")
        self.assertEqual(reasons["is"], "copula")
        self.assertEqual(reasons["at"], "preposition")

    def test_multiword_expressions(self):
        self.assertEqual(gloss("I love you"), "I-LOVE-YOU")
        self.assertEqual(gloss("Thank you very much."), "THANK-YOU")

    def test_proper_nouns_and_unknown_words_are_fingerspelled(self):
        self.assertEqual(gloss("My name is Priya."), "MY NAME fs-PRIYA")
        out = translate("I like pizza.")
        self.assertIn("fs-PIZZA", out["sentences"][0]["gloss"])
        self.assertEqual(out["stats"]["fingerspelled"], 1)

    def test_numbers(self):
        self.assertEqual(gloss("She has 3 cats."), "IX-3 HAVE THREE CAT")
        self.assertEqual(gloss("I have two books."), "ME HAVE TWO BOOK")

    def test_contractions(self):
        self.assertEqual(gloss("I don't understand."), "ME NOT UNDERSTAND")
        self.assertEqual(gloss("I can't go."), "ME CAN NOT GO")
        self.assertEqual(gloss("What's your name?"), "YOUR NAME WHAT")

    def test_her_possessive_vs_object(self):
        self.assertEqual(gloss("Her name is Anna."), "POSS-3 NAME fs-ANNA")
        self.assertEqual(gloss("I love her."), "ME LOVE IX-3")


class NonManualMarkers(unittest.TestCase):
    def nmm(self, text):
        return [(n["type"], n["from"], n["to"]) for n in sentence(text)["nmm"]]

    def test_wh_question_furrows_brows(self):
        self.assertEqual(self.nmm("What is your name?"), [("wh", 0, 2)])

    def test_yes_no_question_raises_brows(self):
        self.assertEqual(sentence("Do you want water?")["type"], "yn")
        self.assertEqual(self.nmm("Do you want water?"), [("yn", 0, 2)])

    def test_question_marker_only_spans_question_clause(self):
        self.assertEqual(self.nmm("Hello, what is your name?"), [("wh", 1, 3)])

    def test_negation_headshake_spans_to_clause_end(self):
        self.assertEqual(self.nmm("He is not my friend."), [("neg", 1, 3)])


class Plan(unittest.TestCase):
    def test_plan_items_reference_lexicon_or_fingerspelling(self):
        signs = load_json("lexicon.json")["signs"]
        out = translate("Hello, my name is Zed. What is your name? I don't understand.")
        kinds = [i["kind"] for i in out["plan"]["items"]]
        self.assertEqual(kinds.count("pause"), 2)
        for item in out["plan"]["items"]:
            if item["kind"] == "sign":
                self.assertIn(item["id"], signs)
            elif item["kind"] == "fs":
                self.assertTrue(item["word"].isalnum())

    def test_empty_input(self):
        out = translate("   ")
        self.assertEqual(out["plan"]["items"], [])


class LexiconIntegrity(unittest.TestCase):
    """Static checks; node static/text2sign/tools/validate.mjs does the 3D ones."""

    @classmethod
    def setUpClass(cls):
        cls.lex = load_json("lexicon.json")["signs"]
        cls.shapes = load_json("handshapes.json")["shapes"]
        with open(os.path.join(ROOT, "static", "text2sign", "avatar.js")) as f:
            src = f.read()
        block = src[src.index("export const LOCATIONS"):src.index("};", src.index("export const LOCATIONS"))]
        cls.locations = set(re.findall(r"^\s+(\w+):", block, re.M))

    def frames(self):
        for sid, sign in self.lex.items():
            for side in ("r", "l"):
                frames = sign.get(side)
                if isinstance(frames, list):
                    for f in frames:
                        yield sid, side, f

    def test_handshapes_exist(self):
        for sid, side, f in self.frames():
            if "hs" in f:
                self.assertIn(f["hs"], self.shapes, f"{sid}/{side}")

    def test_locations_exist(self):
        for sid, side, f in self.frames():
            if isinstance(f.get("loc"), str):
                self.assertIn(f["loc"], self.locations, f"{sid}/{side}")

    def test_first_frame_fully_specified(self):
        for sid, sign in self.lex.items():
            for side in ("r", "l"):
                frames = sign.get(side)
                if isinstance(frames, list):
                    for key in ("hs", "loc", "palm", "fing"):
                        self.assertIn(key, frames[0], f"{sid}/{side} first frame needs '{key}'")

    def test_every_sign_documented_and_reachable_from_english(self):
        for sid, sign in self.lex.items():
            self.assertTrue(sign.get("desc"), sid)
            self.assertTrue(sign.get("words"), sid)

    def test_word_lists_do_not_collide(self):
        seen = {}
        for sid, sign in self.lex.items():
            for w in sign["words"]:
                self.assertNotIn(w, seen, f"'{w}' maps to both {seen.get(w)} and {sid}")
                seen[w] = sid


if __name__ == "__main__":
    unittest.main()
