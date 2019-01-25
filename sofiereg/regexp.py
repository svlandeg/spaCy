import re
import regex


def regtest1():
    # variable look-forward
    text1 = "example test to match 100 doooollar"
    reg1 = re.compile(r"\d+ (?=(do*llar)|(pesos))")
    print(reg1.search(text1))


def regtest2():
    # variable look-behind: look-behind requires fixed-width pattern --> does not work
    text2 = "that would be USSSD100"
    reg2 = regex.compile(r"(?<=US*D)\d+")
    # reg2 = re.compile(r"(?<=US*D)\d+")
    print(reg2.search(text2))


if __name__ == "__main__":
    regtest1()
    regtest2()
