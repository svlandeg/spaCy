import re
import regex


def search_re(text, reg):
    print(re.compile(reg).search(text))


def search_regex(text, reg):
    print(regex.compile(reg).search(text))


if __name__ == "__main__":

    # variable look-forward
    search_re("example test to match 100 doooollar", r"\d+ (?=(do*llar)|(pesos))")

    # variable look-behind: look-behind requires fixed-width pattern --> does not work with re
    # search_re("that would be USSSD100", r"(?<=US*D)\d+")

    # variable look-behind: --> works with regexp
    search_regex("that would be USSSD100", r"(?<=US*D)\d+")
