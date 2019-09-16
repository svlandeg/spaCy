# coding: utf-8
from __future__ import unicode_literals

from spacy.tokens import Span

"""Script that evaluates the influence of coref annotations on the performance
of an entity linking model.
"""

import datetime
from pathlib import Path

import spacy

# TODO: clean up paths
from bin.wiki_entity_linking import training_set_creator
from bin.wiki_entity_linking.wikidata_train_entity_linker import measure_baselines, measure_acc
from spacy.kb import KnowledgeBase

# wp_train_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_small/")
wp_train_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_train_dataset/")
kb_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_KB/")
nlp_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_EL/nlp/")

dev_limit = None


def now():
    return datetime.datetime.now()


class CorefComponent(object):
    def __init__(self):
        Span.set_extension("coref_cluster", default=[])

    def __call__(self, doc):
        for ent in doc.ents:
            # TODO
            ent._.coref_cluster = []
        return doc


    def add_coref_from_file(self, data, coref_data_by_article):
        """ Set coref annotations from file"""
        # TODO: this is a toy method, should be done by neuralcoref directly!

        for doc, gold in data:
            article_id = doc.user_data["orig_article_id"]
            coref_in_doc = coref_data_by_article.get(article_id, dict())

            for cluster_id, span_list in coref_in_doc.items():
                for ent in span_list:
                    ent._.coref_cluster = span_list
                    # print("SET", span_list, "for", ent)


def eval_wp():
    # STEP 1 : load the NLP object
    print()
    print(now(), "STEP 1: loading model from", nlp_dir)
    nlp = spacy.load(nlp_dir)
    # adding toy coref component to the cluster
    coref_comp = CorefComponent()
    nlp.add_pipe(coref_comp, after="ner")  # add it to the pipeline

    # STEP 2 : read the KB
    print()
    print(now(), "STEP 2: reading the KB from", kb_dir / "kb")
    kb = KnowledgeBase(vocab=nlp.vocab)
    kb.load_bulk(kb_dir / "kb")

    print()
    print(now(), "STEP 3: reading the training data from", wp_train_dir)
    wp_data, coref_data_by_article = training_set_creator.read_training(
        nlp=nlp,
        training_dir=wp_train_dir,
        dev=True,
        limit=dev_limit,
        kb=None,
        sentence=False,
        coref=True,
    )
    print("Dev testing on", len(wp_data), "docs")
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    print()
    print(now(), "STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(wp_data, kb, nlp)

    print()
    print(now(), "STEP 5: set coref annotations from file")
    coref_comp.add_coref_from_file(wp_data, coref_data_by_article)
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    print()
    print(now(), "STEP 6: measuring the baselines and EL performance of dev data")
    measure_performance(wp_data, kb, nlp)

    # print()
    # print(now(), "STEP 6: adjust predictions to follow coref chain")
    # new_data = coref_global_optimum(data_per_sentence, coref_data_by_article, nlp)

    # print()
    # print(now(), "STEP 7: measuring the baselines and EL performance of NEW dev data")
    # measure_performance(new_data, kb, nlp)


def measure_performance(data, kb, nlp):
    counts, acc_r, acc_r_d, acc_p, acc_p_d, acc_o, acc_o_d = measure_baselines(data, kb)
    print("dev counts:", sorted(counts.items(), key=lambda x: x[0]))

    oracle_by_label = [(x, round(y, 3)) for x, y in acc_o_d.items()]
    print("dev accuracy oracle:", round(acc_o, 3), oracle_by_label)

    random_by_label = [(x, round(y, 3)) for x, y in acc_r_d.items()]
    print("dev accuracy random:", round(acc_r, 3), random_by_label)

    prior_by_label = [(x, round(y, 3)) for x, y in acc_p_d.items()]
    print("dev accuracy prior:", round(acc_p, 3), prior_by_label)

    el_pipe = nlp.get_pipe("entity_linker")
    # using only context
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = False
    dev_acc_context, dev_acc_cont_d = measure_acc(data, el_pipe)
    context_by_label = [(x, round(y, 3)) for x, y in dev_acc_cont_d.items()]
    print("dev accuracy context:", round(dev_acc_context, 3), context_by_label)

    # measuring combined accuracy (prior + context)
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = True
    dev_acc_combo, dev_acc_combo_d = measure_acc(data, el_pipe)
    combo_by_label = [(x, round(y, 3)) for x, y in dev_acc_combo_d.items()]
    print("dev accuracy prior+context:", round(dev_acc_combo, 3), combo_by_label)


if __name__ == "__main__":
    eval_wp()
