# coding: utf8
from __future__ import unicode_literals
from spacy.lang.en import English

import spacy

from spacy.symbols import ORTH, POS, X, LEMMA, TAG, NOUN, IS_PUNCT


def test_issue3944():
    """Test that adding a special case to Tokenizer works properly
    """
    nlp = spacy.load("en_core_web_md")

    nlp.tokenizer.add_special_case('»', [{ORTH: '»', POS: X, TAG: X, LEMMA: 'Test'}])
    nlp.tokenizer.add_special_case('lying', [{ORTH: 'lying', POS: X, TAG: X, LEMMA: 'Test'}])

    doc = nlp('He said: »I am lying.«')

    print()
    for token in doc:
        print('{:10}{:10}{:10}{:10}'.format(token.text, token.pos_, token.tag_, token.lemma_))
