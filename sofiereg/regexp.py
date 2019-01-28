import re
import regex
import spacy

unicode_path = spacy.util.ensure_path("C:/Users/Sofie/Documents/git/spacy/sofiereg/UniCodeData.txt")


def search_re(text, reg):
    print(re.compile(reg).search(text))


def search_regex(text, reg):
    print(regex.compile(reg).search(text))


def find_unicode(code):
    results = []
    with unicode_path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            words = line.split(';')
            if words[2] == code:
                results.append(words[0])
    return results


def get_char_list(unicode_list):
    return "[" + "".join(chr(int(code, 16)) for code in unicode_list) + "]"


def get_char_string(unicode_list):
    return " ".join(chr(int(code, 16)) for code in unicode_list)


if __name__ == "__main__":

    # variable look-forward
    # search_re("example test to match 100 doooollar", r"\d+ (?=(do*llar)|(pesos))")

    # variable look-behind: look-behind requires fixed-width pattern --> does not work with re
    # search_re("that would be USSSD100", r"(?<=US*D)\d+")

    # variable look-behind: --> works with regexp
    # search_regex("that would be USSSD100", r"(?<=US*D)\d+")

    search_regex("can you still dunk?🍕🍔😵LOL", r"[\p{So}]")
    search_regex("i💙you", r"[\p{So}]")
    search_regex("🤘🤘yay!", r"[\p{So}]")

    unicode_match = get_char_list(find_unicode("So"))

    search_re("can you still dunk?🍕🍔😵LOL°", unicode_match)
    search_re("i💙you", unicode_match)
    search_re("🤘🤘yay!", unicode_match)
