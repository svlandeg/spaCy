# coding: utf8
from __future__ import unicode_literals

from spacy.util import minibatch

from spacy.lang.en import English


def test_issue3912():
    """Test that textcat fails more gracefully when initializing wrongly"""
    import spacy
    nlp = spacy.load("de_core_news_sm")

    x_train = ["I am a colored example", "I have no color at all", "I don't think I have any either"]
    y_train = [{"COLOR": 1}, {"COLOR": 0}, {"COLOR": 0}]

    train_data = list(zip(x_train, [{'cats': cats} for cats in y_train]))
    print(train_data)

    en = English()
    textcat = nlp.create_pipe(
        "textcat", config={"exclusive_classes": False, "architecture": "bow" })
    # textcat.add_label("COLOR")   # TODO: friendly error when this line is gone

    nlp.add_pipe(textcat, last=True)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for i in range(2):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data)

            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(docs=texts, golds=annotations, sgd=optimizer, drop=0.1, losses=losses)

    print(textcat.model)
    textcat.model.use_params(optimizer.averages)