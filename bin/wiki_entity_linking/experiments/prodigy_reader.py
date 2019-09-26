# coding: utf-8
from __future__ import unicode_literals

from pathlib import Path

import srsly

from spacy.gold import GoldParse
from spacy.lang.en import English

prodigy_eval_dir = Path(
    "C:/Users/Sofie/Documents/git/prodigy-recipes/entity_linker/eval/data"
)
wp_data = "eval_wp_el_output_310.jsonl"


class NewsParser:
    # news_data = "annotate_news_output_350.jsonl"
    news_data = "annotate_news_output_500.jsonl"
    nil_news_data = "annotate_news_nil_output_226.jsonl"
    qids = 0
    duplicates = 0
    invalid = 0
    nils = dict()
    invalid_labels = 0
    LABELS_TO_IGNORE = ['CARDINAL', 'DATE', 'MONEY', 'ORDINAL', 'QUANTITY', 'TIME', 'PERCENT']

    def read_news_data(self, nlp, orig=True, free_text=True):
        data = []
        jsons_by_article = dict()
        nil_jsons_by_article = dict()

        json_loc_1 = prodigy_eval_dir / self.news_data
        json_loc_2 = prodigy_eval_dir / self.nil_news_data

        if orig:
            # collect all original annotations per article ID
            for json_obj in srsly.read_jsonl(json_loc_1):
                article_id = int(json_obj["article_id"])
                other_objs = jsons_by_article.get(article_id, list())
                other_objs.append(json_obj)
                jsons_by_article[article_id] = other_objs

            # process each article
            for article_id, json_objs in jsons_by_article.items():
                data.extend(self.process_article(nlp, article_id, json_objs))

        if free_text:
            # collect all NIL free-text annotations per article ID
            for json_obj in srsly.read_jsonl(json_loc_2):
                article_id = int(json_obj["article_id"])
                other_objs = nil_jsons_by_article.get(article_id, list())
                other_objs.append(json_obj)
                nil_jsons_by_article[article_id] = other_objs

            # process each article with NIL annotations
            for article_id, json_objs in nil_jsons_by_article.items():
                data.extend(self.process_article_nil(nlp, article_id, json_objs))

        print("Q IDs:", self.qids)
        print("parsed articles:", len(jsons_by_article.keys()))
        print("duplicate annotations:", self.duplicates)
        print("invalid annotations:", self.invalid)
        print("invalid labels:", self.invalid_labels)
        print("data size (# articles):", len(data))
        print("data size (# links):", sum([len(gold.links) for doc, gold in data]))
        for nil, count in self.nils.items():
            print(nil, ":", count)
        print()

        return data

    def process_article(self, nlp, article_id, json_objs):
        # here we assume that all json_objs are from the same article
        if not json_objs:
            return []

        article = json_objs[0]["article_text"]
        article_doc = nlp(article)
        article_doc.user_data["orig_article_id"] = article_id

        gold_entities = {}

        for json_obj in json_objs:
            assert article_id == int(json_obj["article_id"])
            # sentence = json_obj["text"]
            sent_offset = int(json_obj["sent_offset"])
            answer = json_obj["answer"]
            accept = json_obj["accept"]
            if answer.strip() != "accept" or len(accept) != 1:
                self.invalid += 1
            else:
                gold_id = accept[0].strip()
                spans = json_obj["spans"]
                assert len(spans) == 1
                span = spans[0]
                if span["label"].strip() in self.LABELS_TO_IGNORE:
                    self.invalid_labels += 1
                elif gold_id.startswith("NIL_"):
                    previous_count = self.nils.get(gold_id, 0)
                    previous_count += 1
                    self.nils[gold_id] = previous_count
                else:
                    start = int(span["start"])
                    end = int(span["end"])
                    tuple_key = (start + sent_offset, end + sent_offset)
                    # we found a duplicate annotation - ignore
                    if gold_entities.get(tuple_key, None):
                        assert list(gold_entities[tuple_key].keys()) == [gold_id]
                        self.duplicates += 1
                    else:
                        gold_entities[tuple_key] = {gold_id: 1.0}
                        self.qids += 1

        if not gold_entities:
            return []

        goldparse = GoldParse(doc=article_doc, links=gold_entities)
        return [(article_doc, goldparse)]

    def process_article_nil(self, nlp, article_id, json_objs):
        # here we assume that all json_objs are from the same article
        if not json_objs:
            return []

        article = json_objs[0]["article_text"]
        article_doc = nlp(article)
        article_doc.user_data["orig_article_id"] = article_id

        gold_entities = {}

        for json_obj in json_objs:
            assert article_id == int(json_obj["article_id"])
            answer = json_obj["answer"]
            gold_id = json_obj.get("user_text", "").strip()
            if answer.strip() != "accept" or not (gold_id.startswith("Q") or gold_id.startswith("NIL")):
                self.invalid += 1
            else:
                spans = json_obj["spans"]
                assert len(spans) == 1
                span = spans[0]

                if span["label"] in self.LABELS_TO_IGNORE:
                    self.invalid_labels += 1
                else:
                    if gold_id.startswith("NIL_"):
                        previous_count = self.nils.get(gold_id, 0)
                        previous_count += 1
                        self.nils[gold_id] = previous_count
                    elif gold_id.startswith("Q"):
                        start = int(span["start"])
                        end = int(span["end"])
                        tuple_key = (start, end)
                        # we found a duplicate annotation - ignore
                        if gold_entities.get(tuple_key, None):
                            assert list(gold_entities[tuple_key].keys()) == [gold_id]
                            self.duplicates += 1
                        else:
                            gold_entities[tuple_key] = {gold_id: 1.0}
                            self.qids += 1

        if not gold_entities:
            return []

        goldparse = GoldParse(doc=article_doc, links=gold_entities)
        return [(article_doc, goldparse)]


if __name__ == "__main__":
    news_data = NewsParser().read_news_data(English(), orig=False, free_text=True)
    for (doc, gold) in news_data:
        doc_id = doc.user_data["orig_article_id"]
        print(" - doc", doc_id, len(doc.ents), "entities")
        print(gold.links)
        print()
