# coding: utf-8
from __future__ import unicode_literals

from spacy.tokens import Span


class OfflineCorefComponent(object):
    """ This component is used to load in coref annotations from file (for Wikipedia data) """
    def __init__(self):
        Span.set_extension("coref_cluster", default=[])

    def __call__(self, doc):
        for ent in doc.ents:
            # initialize to empty ent
            ent._.coref_cluster = []
        return doc

    @staticmethod
    def add_coref_from_file(data, coref_data_by_article):
        """ Set coref annotations from file"""
        # TODO: this is a toy method, should be done by neuralcoref directly!

        for doc, gold in data:
            article_id = doc.user_data["orig_article_id"]
            coref_in_doc = coref_data_by_article.get(article_id, dict())

            for cluster_id, span_list in coref_in_doc.items():
                for ent in span_list:
                    ent._.coref_cluster = span_list
                    # print("SET", span_list, "for", ent)
