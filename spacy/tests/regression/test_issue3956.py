# coding: utf8
from __future__ import unicode_literals

import spacy


def test_issue3956():
    nlp = spacy.load("en_core_web_sm")

    sentences = [
        "Fruits, for example bananas, oranges and peaches.",
        "Fruits, for example, bananas, oranges and peaches.",
        "Fruits, for example apples, bananas, oranages and peaches.",
        "Fruits, for example, apples, bananas, oranges and peaches.",
        "Fruits, for example apples, bananas, watermelons, oranages and peaches.",
        "Fruits, for example, apples, bananas, watermelons, oranages and peaches."
    ]

    cnt = 6

    for sentence in sentences:
        doc = nlp(sentence)
        print()
        print(doc)
        for token in doc:
            print(token.text, token.pos_, token.dep_, token.head.text)

        results = list([i for i in doc.sents][0].noun_chunks)
        print(results)
        if cnt & 1:
            print("'for example,' with comma")
        else:
            print("'for example' without comma")

        print("Sentence: {sentence}\nReal Examples: {coe}\nNumber of noun phrases:"
              " {len_of_result}\nResults: {results}\n\n".
             format(sentence=sentence, coe=cnt//2,
                    len_of_result=len(results), results=results))
        cnt += 1
