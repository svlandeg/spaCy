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

wp_train_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_small/")
kb_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_KB/")
nlp_dir = Path("C:/Users/Sofie/Documents/data/spacy_test_CLI_EL/nlp/")

dev_limit = 10


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


def eval_wp():
    # STEP 1 : load the NLP object
    print()
    print(now(), "STEP 1: loading model from", nlp_dir)
    nlp = spacy.load(nlp_dir)
    # adding toy coref component to the cluster
    nlp.add_pipe(CorefComponent(), after="ner")  # add it to the pipeline

    # STEP 2 : read the KB
    print()
    print(now(), "STEP 2: reading the KB from", kb_dir / "kb")
    kb = KnowledgeBase(vocab=nlp.vocab)
    kb.load_bulk(kb_dir / "kb")

    print()
    print(now(), "STEP 3: reading the training data from", wp_train_dir)
    data_per_sentence, coref_data_by_article = training_set_creator.read_training(
        nlp=nlp,
        training_dir=wp_train_dir,
        dev=True,
        limit=dev_limit,
        kb=None,
        sentence=True,
        coref=True,
    )
    print("Dev testing on", len(data_per_sentence), "sentences")

    print()
    print(now(), "STEP 4: measuring the baselines and EL performance of dev data")
    measure_performance(data_per_sentence, kb, nlp)

    # print()
    # print(now(), "STEP 5: adjust predictions to follow coref chain")
    # new_data = coref_global_optimum(data_per_sentence, coref_data_by_article, nlp)

    # print()
    # print(now(), "STEP 6: measuring the baselines and EL performance of NEW dev data")
    # measure_performance(new_data, kb, nlp)


def coref_global_optimum(data, coref_data_by_article, nlp):
    for sentence, gold in data:
        sent_doc = nlp(sentence.text)

        u = sentence.user_data
        article_id = u["orig_article_id"]
        sent_offset = u["sent_offset"]
        coref_doc, coref_in_doc = coref_data_by_article.get(article_id, dict())

        doc_offset_to_cluster_id = dict()
        for cluster_id, span_list in coref_in_doc.items():
            for entity in span_list:
                doc_offset_to_cluster_id[(entity.start_char, entity.end_char)] = cluster_id

        for offset, value in gold.links.items():
            start_char, end_char = offset
            doc_start_char = start_char + sent_offset
            doc_end_char = end_char + sent_offset
            mention = sentence.text[start_char:end_char]
            mention2 = coref_doc.text[start_char+sent_offset:end_char+sent_offset]
            doc_offset = (doc_start_char, doc_end_char)
            print("article/sentence", article_id, sent_offset, "offset", offset, "-->", "value", value, mention, "==", mention2)
            for ent in sent_doc.ents:
                if ent.start_char == start_char:
                    print(" -->", ent, ent.kb_id_)
            cluster_id = doc_offset_to_cluster_id.get(doc_offset, None)
            if cluster_id:
                print(" coref cluster", coref_in_doc[cluster_id])
                for coref_span in coref_in_doc[cluster_id]:
                    if coref_span.start_char != doc_start_char:
                        print(" coref span", coref_span.start_char, coref_span.end_char)
                        coref_span_doc = nlp(coref_span.sent.text)
                        for ent in coref_span_doc.ents:
                            if ent.start_char == coref_span.start_char - coref_span.sent.start_char:
                                print(" -->", ent, ent.kb_id_)

            print()

    return data


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
