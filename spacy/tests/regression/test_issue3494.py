# coding: utf8
from __future__ import unicode_literals

from spacy.matcher import PhraseMatcher
from spacy.lang.en import English
from spacy.compat import pickle


def test_issue3494(en_vocab):
    nlp1 = English()
    matcher = PhraseMatcher(nlp1.vocab)
    matcher.add("TEST1", None, nlp1("a"), nlp1("b"), nlp1("c"))
    matcher.add("TEST2", None, nlp1("d"))

    data = pickle.dump(matcher, open("matcher.pickle", "wb"))

    pickled_matcher = pickle.load(open("matcher.pickle", "rb"))

    print()
    m = matcher(nlp1("a b c"))
    m_pickeld = pickled_matcher(nlp1("a b c"))
    assert m == m_pickeld

    nlp2 = English()
    loaded_matcher = pickle.load(open("matcher.pickle", "rb"))
    m_loaded = loaded_matcher(nlp2("a b c"))
    assert m == m_loaded