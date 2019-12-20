# coding: utf-8
from __future__ import unicode_literals

from bin.wiki_entity_linking.entity_linker_evaluation import measure_performance
from neuralcoref import NeuralCoref

from bin.wiki_entity_linking.experiments.prodigy_reader import NewsParser

"""Script that evaluates the influence of coref annotations on the performance
of an entity linking model.
"""

from pathlib import Path
import logging

import spacy

from bin.wiki_entity_linking import wikipedia_processor
from bin.wiki_entity_linking import TRAINING_DATA_FILE

# TODO: clean up paths
nlp_dir = Path("C:/Users/Sofie/Documents/data/EL-data/RUN_full/nlp/")
training_path = Path("C:/Users/Sofie/Documents/data/EL-data/KB/") / TRAINING_DATA_FILE

logger = logging.getLogger(__name__)


def eval_el():
    # STEP 1 : load the NLP object
    logger.info("STEP 1: loading model from {}".format(nlp_dir))
    nlp = spacy.load(nlp_dir)
    entity_linker = nlp.get_pipe("entity_linker")
    logger.info("Entity linker cfg = {}".format(entity_linker.cfg))

    # STEP 2 : read the KB
    logger.info("STEP 2: reading KB")
    kb = entity_linker.kb
    if kb is None:
        logger.error("KB should not be None")

    coref = True
    if coref:
        logger.info("Adding coreference resolution to the pipeline")
        coref = NeuralCoref(nlp.vocab, name='neuralcoref', greedyness=0.5)
        nlp.add_pipe(coref, before="entity_linker")

    dev_articles = 50
    eval_wp(nlp, kb, dev_articles)
    # eval_news(nlp, kb)
    # eval_toy(nlp)


def eval_news(nlp, kb):
    # STEP 3 : read the dev data
    np = NewsParser()
    orig = True
    free_text = True
    logger.info("STEP 3: reading the dev data, orig={} free_text={}".format(orig, free_text))
    news_data = np.read_news_data(nlp, orig=orig, free_text=free_text)

    logger.info("Dev testing on {} docs".format(len(news_data)))

    # STEP 4 : Measure performance on the dev data
    logger.info("STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(news_data, kb, nlp.get_pipe("entity_linker"))


def eval_toy(nlp):
    text = (
        "The book was written by Douglas Adams. "
        "Adams was a funny man."
    )
    doc = nlp(text)
    for ent in doc.ents:
        print()
        print(["ent", ent.text, ent.label, ent.kb_id_, ])
        if ent.has_extension("coref_cluster"):
            print(ent._.coref_cluster)


def eval_wp(nlp, kb, dev_articles):
    # STEP 3 : read the dev data
    logger.info("STEP 3: reading the dev data from {}".format(training_path))

    train_indices, dev_indices = wikipedia_processor.read_training_indices(training_path)
    if dev_articles:
        dev_indices = dev_indices[0:dev_articles]

    wp_data = wikipedia_processor.read_el_docs_golds(
        nlp=nlp,
        entity_file_path=training_path,
        dev=True,
        line_ids=dev_indices,
        kb=kb,    # TODO: should be None ?
        labels_discard=nlp.get_pipe("entity_linker").cfg.get("labels_discard", [])
    )

    logger.info("Dev testing on {} docs".format(len(dev_indices)))

    # STEP 4 : Measure performance on the dev data
    logger.info("STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(wp_data, kb, nlp.get_pipe("entity_linker"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_el()