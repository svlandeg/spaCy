# coding: utf-8
from __future__ import unicode_literals

from bin.wiki_entity_linking.experiments.coref_comp import OfflineCorefComponent
from bin.wiki_entity_linking.experiments.prodigy_reader import NewsParser
from spacy.gold import GoldParse

"""Script that evaluates the influence of coref annotations on the performance
of an entity linking model.
"""

import datetime
from pathlib import Path

import spacy
import neuralcoref

# TODO: clean up paths
from bin.wiki_entity_linking import training_set_creator
from bin.wiki_entity_linking.wikidata_train_entity_linker import measure_baselines, measure_acc
from spacy.kb import KnowledgeBase

# wp_train_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_small/")
wp_train_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_train_dataset/")
kb_dir = Path("C:/Users/Sofie/Documents/data/EL-data/KB/")
nlp_dir = Path("C:/Users/Sofie/Documents/data/EL-data/EL/nlp/")



def now():
    return datetime.datetime.now()


def eval_el():
    # STEP 1 : load the NLP object
    print()
    print(now(), "STEP 1: loading model from", nlp_dir)
    nlp = spacy.load(nlp_dir)

    # STEP 2 : read the KB
    print()
    print(now(), "STEP 2: reading the KB from", kb_dir / "kb")
    kb = KnowledgeBase(vocab=nlp.vocab)
    kb.load_bulk(kb_dir / "kb")

    # eval_wp(nlp, kb)
    eval_news(nlp, kb)


def eval_news(nlp, kb):
    # STEP 3 : read the dev data
    np = NewsParser()
    print()
    orig = True
    free_text = True
    print(now(), "STEP 3: reading the dev data, orig", orig, "free_text", free_text)
    news_data = np.read_news_data(nlp, orig=orig, free_text=free_text)

    print("Dev testing on", len(news_data), "docs")
    # for doc, gold in news_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 4 : Measure performance on the dev data
    print()
    print(now(), "STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(news_data, kb, nlp)

    # STEP 5 : set coref annotations with neuralcoref
    print()
    print(now(), "STEP 5: set coref annotations with neuralcoref")
    neuralcoref.add_to_pipe(nlp)
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
    print()
    print(now(), "STEP 6: measuring the baselines and EL performance of dev data + coref")
    measure_performance(news_data_coref, kb, nlp)


def eval_wp(nlp, kb):
    dev_wp_limit = 5

    # STEP 3 : read the dev data
    print()
    print(now(), "STEP 3: reading the dev data from", wp_train_dir)
    wp_data, coref_data_by_article = training_set_creator.read_training(
        nlp=nlp,
        training_dir=wp_train_dir,
        dev=True,
        limit=dev_wp_limit,
        kb=None,
        sentence=False,
        coref=True,
    )
    print("Dev testing on", len(wp_data), "docs")
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 4 : Measure performance on the dev data
    print()
    print(now(), "STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(wp_data, kb, nlp)

    # STEP 5 : set coref annotations from file
    print()
    print(now(), "STEP 5: set coref annotations from file")

    # adding toy coref component to the cluster
    coref_comp = OfflineCorefComponent()
    nlp.add_pipe(coref_comp, after="ner", name="coref")
    coref_comp.add_coref_from_file(wp_data, coref_data_by_article)
    # for doc, gold in wp_data:
        # article_id = doc.user_data["orig_article_id"]
        # print(" - doc", article_id, len(doc.ents), "entities")

    # STEP 6 : measure performance again
    print()
    print(now(), "STEP 6: measuring the baselines and EL performance of dev data + coref")
    measure_performance(wp_data, kb, nlp)


def measure_performance(data, kb, nlp):
    counts, acc_r, acc_r_d, acc_p, acc_p_d, acc_o, acc_o_d = measure_baselines(data, kb)
    print()
    print(" dev counts:", sorted(counts.items(), key=lambda x: x[0]))
    print()

    oracle_by_label = [(x, round(y, 3)) for x, y in acc_o_d.items()]
    print(" dev accuracy oracle:", round(acc_o, 3), oracle_by_label)
    print()

    random_by_label = [(x, round(y, 3)) for x, y in acc_r_d.items()]
    print(" dev accuracy random:", round(acc_r, 3), random_by_label)
    print()

    prior_by_label = [(x, round(y, 3)) for x, y in acc_p_d.items()]
    print(" dev accuracy prior:", round(acc_p, 3), prior_by_label)
    print()

    el_pipe = nlp.get_pipe("entity_linker")
    # using only context
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = False
    dev_acc_context, dev_acc_cont_d = measure_acc(data, el_pipe)
    context_by_label = [(x, round(y, 3)) for x, y in dev_acc_cont_d.items()]
    print(" dev accuracy context:", round(dev_acc_context, 3), context_by_label)
    print()

    # measuring combined accuracy (prior + context)
    el_pipe.cfg["incl_context"] = True
    el_pipe.cfg["incl_prior"] = True
    dev_acc_combo, dev_acc_combo_d = measure_acc(data, el_pipe)
    combo_by_label = [(x, round(y, 3)) for x, y in dev_acc_combo_d.items()]
    print(" dev accuracy prior+context:", round(dev_acc_combo, 3), combo_by_label)
    print()


if __name__ == "__main__":
    eval_el()
