# coding: utf-8
from __future__ import unicode_literals

from pathlib import Path

import srsly

from spacy.gold import GoldParse
from spacy.lang.en import English

prodigy_eval_dir = Path("C:/Users/Sofie/Documents/git/prodigy-recipes/entity_linker/eval/data")
wp_data = "eval_wp_el_output_310.jsonl"


class NewsParser:
    # news_data = "annotate_news_output_350.jsonl"
    news_data = "annotate_news_output_500.jsonl"
    invalid = 0
    nils = dict()

    def read_news_data(self, nlp):
        data = []
        jsons_by_article = dict()

        json_loc = prodigy_eval_dir / self.news_data

        # collect all annotations per article ID
        for json_obj in srsly.read_jsonl(json_loc):
            article_id = int(json_obj["article_id"])
            other_objs = jsons_by_article.get(article_id, list())
            other_objs.append(json_obj)
            jsons_by_article[article_id] = other_objs

        # process each article
        for article_id, json_objs in jsons_by_article.items():
            data.extend(self.process_article(nlp, article_id, json_objs))

        print("parsed articles:", len(jsons_by_article.keys()))
        print("invalid annotations:", self.invalid)
        print("data size:", len(data))
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
            sentence = json_obj["text"]
            sent_offset = int(json_obj["sent_offset"])
            answer = json_obj["answer"]
            accept = json_obj["accept"]
            if answer != "accept" or len(accept) != 1:
                self.invalid += 1
            else:
                gold_id = accept[0]
                if gold_id.startswith("NIL_"):
                    previous_count = self.nils.get(gold_id, 0)
                    previous_count += 1
                    self.nils[gold_id] = previous_count
                else:
                    spans = json_obj["spans"]
                    assert len(spans) == 1
                    span = spans[0]
                    start = int(span["start"])
                    end = int(span["end"])
                    mention = span["text"]
                    label = span["label"]
                    # print(start, end, label, mention, "=", article[sent_offset+start:sent_offset+end], "=", sentence[start:end])
                    gold_entities[(start+sent_offset, end+sent_offset)] = {gold_id: 1.0}

        if not gold_entities:
            return []

        goldparse = GoldParse(doc=article_doc, links=gold_entities)
        return [(article_doc, goldparse)]


if __name__ == "__main__":
    news_data = NewsParser().read_news_data(English())
    for (doc, gold) in news_data:
        doc_id = doc.user_data["orig_article_id"]
        print(" - doc", doc_id, len(doc.ents), "entities")
        print(gold.links)
        print()
