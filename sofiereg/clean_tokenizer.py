import re
import spacy
import time
from spacy.tokenizer import Tokenizer
from spacy.cli.ud.run_eval import run_pairwise_tokenizer_evals

prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
infix_re = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')


class FastTokenizer(object):

    @staticmethod
    def custom_tokenizer():
        return Tokenizer(spacy.blank("en").vocab, prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=simple_url_re.match)


if __name__ == "__main__":
    text = u"This is a first sentence...... And this is another one. Great!"
    treebank_txt_path = spacy.util.ensure_path("C:/Users/Sofie/Documents/data/UD_2_3/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.txt")
    out_path = spacy.util.ensure_path("C:/Users/Sofie/Desktop/test_tokenizer.csv")

    # normal en model
    loading_start_1 = time.time()
    nlp_1 = spacy.blank("en")
    nlp_1.add_pipe(nlp_1.create_pipe('sentencizer'))
    loading_end_1 = time.time()
    loading_time_1 = loading_end_1 - loading_start_1

    doc1 = nlp_1(text)
    print()
    print("normal:\t", [t.text for t in doc1])

    # en model with custom tokenizer
    loading_start_2 = time.time()
    nlp_2 = spacy.blank("en")
    nlp_2.add_pipe(nlp_2.create_pipe('sentencizer'))
    nlp_2.tokenizer = FastTokenizer.custom_tokenizer()
    loading_end_2 = time.time()
    loading_time_2 = loading_end_2 - loading_start_2

    doc2 = nlp_2(text)
    print()
    print("custom:\t", [t.text for t in doc2])

    # pairwise comparison
    model_1 = (nlp_1, loading_time_1, "blank en")
    model_2 = (nlp_2, loading_time_2, "custom en tokenizer")
    print()

    with out_path.open(mode='a', encoding='utf-8') as out_file:
        results_1, results_2 = run_pairwise_tokenizer_evals(model_1, model_2, treebank_txt_path, out_file)
        print()
        print("results 1", results_1)
        print("results 2", results_2)
