# coding: utf8
from __future__ import unicode_literals

import spacy


def test_issue3988():
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_md")

    if "September" in nlp.vocab.strings:
        print("September in StringStore")  # Working as expected

    dog = nlp.vocab.strings["dog"]
    # assert dog not in nlp.vocab  # Working as expected

    september = nlp.vocab.strings["September"]
    assert "September" in nlp.vocab  # AssertionError
    assert september in nlp.vocab  # AssertionError
