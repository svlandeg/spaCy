# coding: utf-8
from __future__ import unicode_literals

from neuralcoref import NeuralCoref

from bin.wiki_entity_linking.experiments.prodigy_reader import NewsParser
from spacy.gold import GoldParse

"""Script that evaluates the influence of coref annotations on the performance
of an entity linking model.
"""

from pathlib import Path
import logging

import spacy

from bin.wiki_entity_linking import wikipedia_processor
from bin.wiki_entity_linking import TRAINING_DATA_FILE
from bin.wiki_entity_linking.entity_linker_evaluation import measure_baselines, get_eval_results

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

    dev_wp_limit = 1000  # TODO: merge PR 4811 and define by # of articles
    eval_wp(nlp, kb, dev_wp_limit, coref=True)
    # eval_news(nlp, kb)
    # eval_toy(nlp, kb)


def eval_news(nlp, kb, coref=False):
    if coref:
        logger.info("Adding coreference resolution to the pipeline")
        coref = NeuralCoref(nlp.vocab, name='neuralcoref', greedyness=0.5)
        nlp.add_pipe(coref, before="entity_linker")

    # STEP 3 : read the dev data
    np = NewsParser()
    orig = True
    free_text = True
    logger.info("STEP 3: reading the dev data, orig {} free_text".format(orig, free_text))
    news_data = np.read_news_data(nlp, orig=orig, free_text=free_text)

    logger.info("Dev testing on {} docs".format(len(news_data)))
    # for doc, gold in news_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 4 : Measure performance on the dev data
    logger.info("STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(news_data, kb, nlp)

    # STEP 5 : set coref annotations with neuralcoref TODO
    logger.info("STEP 5: set coref annotations with neuralcoref")
    # neuralcoref.add_to_pipe(nlp)
    news_data_coref = []
    for doc, gold in news_data:
        coref_doc = nlp(doc.text)
        coref_doc.user_data["orig_article_id"] = doc.user_data["orig_article_id"]
        coref_gold = GoldParse(doc=coref_doc, links=gold.links)
        news_data_coref.extend([(coref_doc, coref_gold)])

    for doc, gold in news_data_coref:
        article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")
        for key, value in gold.links.items():
            start, end = key
            # print(key, doc[start:end], "->", value)

    # STEP 6 : measure performance again
    logger.info("STEP 6: measuring the baselines and EL performance of dev data + coref")
    measure_performance(news_data_coref, kb, nlp)


def eval_toy(nlp, kb):
    text = (
        "The book was written by Douglas Adams. "
        "Adams was a funny man."
    )

    coref = NeuralCoref(nlp.vocab, name='neuralcoref', greedyness=0.5)
    nlp.add_pipe(coref, before="entity_linker")

    doc = nlp(text)

    for ent in doc.ents:
        print(["ent", ent.text, ent.label, ent.kb_id_, ent._.coref_cluster])


def eval_wp(nlp, kb, dev_wp_limit, coref=False):
    if coref:
        logger.info("Adding coreference resolution to the pipeline")
        coref = NeuralCoref(nlp.vocab, name='neuralcoref', greedyness=0.5)
        nlp.add_pipe(coref, before="entity_linker")

    # STEP 3 : read the dev data
    logger.info("STEP 3: reading the dev data from {}".format(training_path))
    wp_data = wikipedia_processor.read_training(
        nlp=nlp,
        entity_file_path=training_path,
        dev=True,
        limit=dev_wp_limit,
        kb=None,
        labels_discard=nlp.get_pipe("entity_linker").cfg.get("labels_discard", []),
        # sentence=False,
        # coref=True,
    )
    logger.info("Dev testing on {} docs".format(len(wp_data)))
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 4 : Measure performance on the dev data
    logger.info("STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(wp_data, kb, nlp)

    # STEP 5 : set coref annotations from file
    # logger.info("STEP 5: set coref annotations from file")

    # TODO fix this code
    # adding toy coref component to the cluster
    # coref_comp = OfflineCorefComponent()
    # nlp.add_pipe(coref_comp, after="ner", name="coref")
    # coref_comp.add_coref_from_file(wp_data, coref_data_by_article)
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 6 : measure performance again
    # logger.info("STEP 6: measuring the baselines and EL performance of dev data + coref")
    # measure_performance(wp_data, kb, nlp)


def measure_performance(data, kb, nlp):
    baseline_accuracies, counts = measure_baselines(data, kb)
    logger.info("Counts: {}".format({k: v for k, v in sorted(counts.items())}))
    logger.info(baseline_accuracies.report_performance("random"))
    logger.info(baseline_accuracies.report_performance("prior"))
    logger.info(baseline_accuracies.report_performance("oracle"))

    el_pipe = nlp.get_pipe("entity_linker")
    # using only context
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = False
    results = get_eval_results(data, el_pipe)
    logger.info(results.report_metrics("context only"))

    # measuring combined accuracy (prior + context)
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = True
    results = get_eval_results(data, el_pipe)
    logger.info(results.report_metrics("context and prior"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_el()
