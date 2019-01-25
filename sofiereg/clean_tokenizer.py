import re
import spacy
import time

from spacy import util
from spacy.tokenizer import Tokenizer
from spacy.cli.ud.run_eval import run_pairwise_tokenizer_evals
from spacy.lang.tokenizer_exceptions import TOKEN_MATCH, BASE_EXCEPTIONS
from spacy.lang.punctuation import TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from spacy.lang.punctuation import TOKENIZER_INFIXES


class FastTokenizer(object):

    def custom_tokenizer(self, lang):
        return self.custom_tokenizer_util_v2(lang)
        # return self.custom_tokenizer_util_v1(lang)
        # return self.custom_tokenizer_clean(lang)

    def custom_tokenizer_clean(self, lang):
        return Tokenizer(spacy.blank(lang).vocab,
                         prefix_search=re.compile(r'''^[\[\("']''').search,
                         suffix_search=re.compile(r'''[\]\)"']$''').search,
                         infix_finditer=re.compile(r'''[-~]''').finditer,
                         token_match=re.compile(r'''^https?://''').match)

    def custom_tokenizer_util_v2(self, lang):
        # mimics code in BaseDefaults.create_tokenizer + added BASE_EXCEPTIONS
        return Tokenizer(spacy.blank(lang).vocab,
                         prefix_search=util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)).search,
                         suffix_search=util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)).search,
                         infix_finditer=util.compile_infix_regex(tuple(TOKENIZER_INFIXES)).finditer,
                         token_match=TOKEN_MATCH,
                         rules=BASE_EXCEPTIONS)

    def custom_tokenizer_util_v1(self, lang):
        # mimics code in BaseDefaults.create_tokenizer
        return Tokenizer(spacy.blank(lang).vocab,
                         prefix_search=util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)).search,
                         suffix_search=util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)).search,
                         infix_finditer=util.compile_infix_regex(tuple(TOKENIZER_INFIXES)).finditer,
                         token_match=TOKEN_MATCH)


if __name__ == "__main__":
    text = u"This is a first sentence...... And this is another one. Great! :)"
    treebank_txt_path = spacy.util.ensure_path("C:/Users/Sofie/Documents/data/UD_2_3/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.txt")
    out_path = spacy.util.ensure_path("C:/Users/Sofie/Desktop/test_tokenizer.csv")

    # normal en model
    name_1 = "blank xx"
    loading_start_1 = time.time()
    nlp_1 = spacy.blank("xx")
    nlp_1.add_pipe(nlp_1.create_pipe('sentencizer'))
    loading_end_1 = time.time()
    loading_time_1 = loading_end_1 - loading_start_1

    doc1 = nlp_1(text)
    print()
    print("normal:\t", [t.text for t in doc1])

    # en model with custom tokenizer
    name_2 = "util xx tokenizer (v2)"
    loading_start_2 = time.time()
    nlp_2 = spacy.blank("xx")
    nlp_2.add_pipe(nlp_2.create_pipe('sentencizer'))
    nlp_2.tokenizer = FastTokenizer().custom_tokenizer("xx")
    loading_end_2 = time.time()
    loading_time_2 = loading_end_2 - loading_start_2

    doc2 = nlp_2(text)
    print()
    print("custom:\t", [t.text for t in doc2])

    # pairwise comparison
    model_1 = (nlp_1, loading_time_1, name_1)
    model_2 = (nlp_2, loading_time_2, name_2)
    print()

    with out_path.open(mode='a', encoding='utf-8') as out_file:
        results_1, results_2 = run_pairwise_tokenizer_evals(model_1, model_2, treebank_txt_path, out_file)
        print()
        print("results 1", results_1)
        print("results 2", results_2)
