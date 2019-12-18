# coding: utf-8
"""Script to create a useful Wikipedia training set, using coreference resolution
to extend the number of training examples. TODO: fit into updated pipeline + document

For the Wikipedia dump: get enwiki-latest-pages-articles-multistream.xml.bz2
from https://dumps.wikimedia.org/enwiki/latest/
"""
from __future__ import unicode_literals

import datetime
from pathlib import Path
import plac
import neuralcoref  # installed from source with pip install -e .

import spacy
from spacy import Errors

from bin.wiki_entity_linking import training_set_creator


def now():
    return datetime.datetime.now()


@plac.annotations(
    dir_kb=("Directory with KB, NLP and related files", "positional", None, Path),
    loc_training=("Directory to hold training data", "positional", None, Path),
    wp_xml=("Path to the downloaded Wikipedia XML dump.", "option", "w", Path),
    skip_raw=("Whether to skip the raw parsing (default False)", "flag", "s", bool),
    coref=("Whether to include coref or not (default False)", "flag", "c", bool),
    parallelize=("Use dask to parallellize work (default False)", "flag", "d", bool),
    limit=("Optionally limit the # of articles read from WP dump", "option", "l", int),
)
def main(
    dir_kb,
    loc_training,
    wp_xml,
    skip_raw=False,
    coref=False,
    parallelize=False,
    limit=None,
):
    print(now(), "Creating training data from Wikipedia")
    print()

    if limit is not None:
        print("Warning: reading only", limit, "Wikipedia articles.")

    # STEP 0: set up IO
    if not loc_training.exists():
        loc_training.mkdir()

    # STEP 1: create a training dataset from WP
    print()
    print(now(), "STEP 1: creating training dataset at", loc_training)

    if skip_raw:
        print(" - skip_raw=True, skipping this step")
    else:
        loc_entity_defs = dir_kb / "entity_defs.csv"
        training_set_creator.create_training(
            wp_input=wp_xml,
            entity_def_input=loc_entity_defs,
            training_output=loc_training,
            limit=limit,
        )

    # STEP 2: add coreference annotations to the training dataset
    print()
    print(now(), "STEP 2: adding coreference annotations to", loc_training)
    if coref:
        nlp_dir = dir_kb / "nlp"
        print(" - loading model from", nlp_dir)
        nlp = spacy.load(nlp_dir)

        # check that there is a NER component in the pipeline
        if "ner" not in nlp.pipe_names:
            raise ValueError(Errors.E152)

        print(" - adding neuralcoref pipe")
        # default max_dist is 50 (small: less accurate & faster)
        neuralcoref.add_to_pipe(nlp, max_dist=20)

        print(" - parallelization with dask:", parallelize)
        # training_set_creator.add_coreference_to_dataset(
        #    nlp=nlp, training_dir=loc_training, parallelize=parallelize
        # )
        training_set_creator.write_coreference_annotations(
            nlp=nlp, training_dir=loc_training, parallelize=parallelize
        )
    else:
        print(" - coref=False, skipping this step")

    print()
    print(now(), "Done")


if __name__ == "__main__":
    plac.call(main)
