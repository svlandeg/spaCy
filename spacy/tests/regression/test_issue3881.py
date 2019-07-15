# coding: utf8
from __future__ import unicode_literals

import spacy


if __name__ == "__main__":
    text = "The increase in non-cash adjustments was primarily due to a " \
           "$26.0 million increase in stock-based compensation as a result of increase in headcount, $15.9 million amortization of issuance cost " \
           "related to convertible notes, and $7.1 million increase in amortization of deferred sales commission and depreciation and amortization."
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(str(text))

    print()
    for i, token in enumerate(doc):
        if token.text == "amortization":
            print(i)
            print([x.text for x in token.ancestors])
            print()

