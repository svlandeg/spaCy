import re
import regex
import spacy
import time

from sofiereg.regexp import search_re, search_regex

from spacy import util
from spacy.tokenizer import Tokenizer
from spacy.cli.ud.run_eval import run_pairwise_tokenizer_evals
from spacy.lang.tokenizer_exceptions import TOKEN_MATCH, BASE_EXCEPTIONS
from spacy.lang.punctuation import TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from spacy.lang.punctuation import TOKENIZER_INFIXES


split_chars = lambda char: list(char.strip().split(" "))


class FastTokenizer(object):

    def __init__(self, lang):
        self.lang = lang

    def custom_tokenizer(self):
        return self.custom_tokenizer()
        # return self.custom_tokenizer_empty()
        # return self.custom_tokenizer_util_v2()
        # return self.custom_tokenizer_util_v1()
        # return self.custom_tokenizer_basic()

    def custom_tokenizer(self):
        print()
        print("TOKENIZER_PREFIXES 1", TOKENIZER_PREFIXES)
        print("TOKENIZER_PREFIXES 2", util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)))
        print(len(TOKENIZER_PREFIXES))
        print()

        print()
        print("TOKENIZER_POSTFIXES 1", TOKENIZER_SUFFIXES)
        print("TOKENIZER_POSTFIXES 2", util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)))
        print(len(TOKENIZER_SUFFIXES))
        print()

        print()
        print("TOKENIZER_INFIXES 1", TOKENIZER_INFIXES)
        print("TOKENIZER_INFIXES 2", util.compile_infix_regex(tuple(TOKENIZER_INFIXES)))
        print(len(TOKENIZER_INFIXES))
        print()

        return Tokenizer(spacy.blank(self.lang).vocab,
                         prefix_search=util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)).search,
                         suffix_search=util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)).search,
                         infix_finditer=util.compile_infix_regex(tuple(TOKENIZER_INFIXES)).finditer,
                         token_match=TOKEN_MATCH,
                         rules=BASE_EXCEPTIONS)

    def custom_tokenizer_util_v2(self):
        # mimics code in BaseDefaults.create_tokenizer + added BASE_EXCEPTIONS
        return Tokenizer(spacy.blank(self.lang).vocab,
                         prefix_search=util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)).search,
                         suffix_search=util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)).search,
                         infix_finditer=util.compile_infix_regex(tuple(TOKENIZER_INFIXES)).finditer,
                         token_match=TOKEN_MATCH,
                         rules=BASE_EXCEPTIONS)

    def custom_tokenizer_util_v1(self):
        # mimics code in BaseDefaults.create_tokenizer
        return Tokenizer(spacy.blank(self.lang).vocab,
                         prefix_search=util.compile_prefix_regex(tuple(TOKENIZER_PREFIXES)).search,
                         suffix_search=util.compile_suffix_regex(tuple(TOKENIZER_SUFFIXES)).search,
                         infix_finditer=util.compile_infix_regex(tuple(TOKENIZER_INFIXES)).finditer,
                         token_match=TOKEN_MATCH)

    def custom_tokenizer_basic(self):
        return Tokenizer(spacy.blank(self.lang).vocab,
                         prefix_search=re.compile(r'''^[\[\("']''').search,
                         suffix_search=re.compile(r'''[\]\)"']$''').search,
                         infix_finditer=re.compile(r'''[-~]''').finditer,
                         token_match=re.compile(r'''^https?://''').match)

    def custom_tokenizer_empty(self):
        return Tokenizer(spacy.blank(self.lang).vocab,
                         prefix_search=re.compile(r'''''').search,
                         suffix_search=re.compile(r'''''').search,
                         infix_finditer=re.compile(r'''''').finditer,
                         token_match=re.compile(r'''''').match)


if __name__ == "__main__":
    text = u"This 'is-a [first] 💙 sen'tence...... And' this–is another— one. Great! :)"
    treebank_txt_path = spacy.util.ensure_path("C:/Users/Sofie/Documents/data/UD_2_3/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.txt")
    out_path = spacy.util.ensure_path("C:/Users/Sofie/Documents/test_tokenizer.csv")
    print_output = True

    # normal en model
    # name_1 = "blank xx"
    # loading_start_1 = time.time()
    # nlp_1 = spacy.blank("xx")
    # nlp_1.add_pipe(nlp_1.create_pipe('sentencizer'))
    # loading_end_1 = time.time()
    # loading_time_1 = loading_end_1 - loading_start_1

    # doc1 = nlp_1(text)
    # print()
    # print("normal:\t", [t.text for t in doc1])

    # en model with custom tokenizer
    name_2 = "empty prefix xx"
    loading_start_2 = time.time()
    nlp_2 = spacy.blank("xx")
    nlp_2.add_pipe(nlp_2.create_pipe('sentencizer'))
    nlp_2.tokenizer = FastTokenizer("xx").custom_tokenizer()
    loading_end_2 = time.time()
    loading_time_2 = loading_end_2 - loading_start_2

    doc2 = nlp_2(text)
    print()
    print("custom:\t", [t.text for t in doc2])

    # pairwise comparison
    if print_output:
        model_1 = None
        model_2 = (nlp_2, loading_time_2, name_2)
        print()

        with out_path.open(mode='a', encoding='utf-8') as out_file:
            results_1, results_2 = run_pairwise_tokenizer_evals(model_1, model_2, treebank_txt_path, out_file)
            print()
            print("results 1", results_1)
            print("results 2", results_2)
