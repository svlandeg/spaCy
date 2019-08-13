# coding: utf-8
"""Script to create a useful Wikipedia training set, using coreference resolution
to extend the number of training examples.

For the Wikipedia dump: get enwiki-latest-pages-articles-multistream.xml.bz2
from https://dumps.wikimedia.org/enwiki/latest/
"""
from __future__ import unicode_literals

import datetime
from pathlib import Path
import plac
import neuralcoref  # installed from source with pip install -e .

import spacy
from spacy.kb import KnowledgeBase
from spacy import Errors

from bin.wiki_entity_linking import training_set_creator


def now():
    return datetime.datetime.now()


@plac.annotations(
    dir_kb=("Directory with KB, NLP and related files", "positional", None, Path),
    loc_training=("Directory to hold training data", "positional", None, Path),
    wp_xml=("Path to the downloaded Wikipedia XML dump.", "positional", None, Path),
    limit=("Optional threshold to limit lines read from WP dump", "option", "l", int),
)
def main(dir_kb, loc_training, wp_xml, limit=None):
    print(now(), "Creating training data from Wikipedia")
    print()

    if limit is not None:
        print("Warning: reading only", limit, "lines of Wikipedia dump.")

    # STEP 0: set up IO
    if not loc_training.exists():
        loc_training.mkdir()

    # STEP 1 : load the NLP object
    nlp_dir = dir_kb / "nlp"
    print(now(), "STEP 1: loading model from", nlp_dir)
    nlp = spacy.load(nlp_dir)
    neuralcoref.add_to_pipe(nlp)
    print(" - added neuralcoref pipe")

    # check that there is a NER component in the pipeline
    if "ner" not in nlp.pipe_names:
        raise ValueError(Errors.E152)

    # STEP 2 : read the KB
    print()
    print(now(), "STEP 2: reading the KB from", dir_kb / "kb")
    kb = KnowledgeBase(vocab=nlp.vocab)
    kb.load_bulk(dir_kb / "kb")

    # STEP 3: create a training dataset from WP
    print()
    print(now(), "STEP 3: creating training dataset at", loc_training)

    loc_entity_defs = dir_kb / "entity_defs.csv"
    training_set_creator.create_training(
        wikipedia_input=wp_xml,
        entity_def_input=loc_entity_defs,
        training_output=loc_training,
        limit=limit,
    )

    # STEP 4: parse the training data
    print()
    print(now(), "STEP 4: parse the training & evaluation data")

    # for training, get pos & neg instances that correspond to entries in the kb
    print("Parsing training data")
    train_data = training_set_creator.read_training(
        nlp=nlp, training_dir=loc_training, dev=False, limit=None, kb=kb, coref=True
    )

    print("Read", len(train_data), "training instances")
    for doc, gold in train_data:
        print()
        print("doc", doc)
        for key, value in gold.links.items():
            print("link", key, '-->', value)
    print()


if __name__ == "__main__":
    plac.call(main)