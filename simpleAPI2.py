import re
from typing import Iterator, List, Set

from nltk import word_tokenize as nltk_word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer as Stemmer
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.util import trigrams  # skipgrams(_, n, k); n - deg, k - skip dist
from pymystem3 import Mystem


class Data_settings:
    def __init__(self, language: str):
        self.language = language.lower()
        self.stopwords = set(stopwords.words(self.language))

    puncts = set('.,!?():;"[]{}/')

    def sent_tokenize(self, text):
        return nltk_sent_tokenize(text, self.language)

    def word_tokenize(self, text):
        return nltk_word_tokenize(text, self.language)


# Remove unnecessary tokens
def remove_sth(seq: Iterator[str], sth: Set[str]) -> Iterator[str]:
    """ Generic function for removal """
    return filter(lambda x: x not in sth, seq)


def remove_puncts(puncts, seq: Iterator[str]) -> Iterator[str]:
    return remove_sth(seq, puncts)


def remove_stops(stopwords, seq: Iterator[str]) -> Iterator[str]:
    return remove_sth(seq, stopwords)


def wordsToStemmed(sent: Iterator[str]) -> List[str]:
    return [Sentence.stemmer.stem(word) for word in sent]


def wordsToLemmed(sent: Iterator[str]) -> List[str]:
    return [Sentence.lemmer.lemmatize(word) for word in sent]


# Kernel classes
class Sentence:
    lemmer = Mystem()
    stemmer = Stemmer()

    def __init__(self, index: int, sent: str, start: int, end: int, settings: Data_settings):
        self.index = index
        self.word_tokenize = settings.word_tokenize
        self.puncts = settings.puncts
        self.language = settings.language
        self.stopwords = settings.stopwords
        self.sent = sent
        self.words = self.sentToWords()
        self.nGrams = list(trigrams(self.words))
        self.start = start
        self.end = end

    def sentToWords(self) -> List[str]:
        if self.language == 'russian':
            return wordsToLemmed(
                remove_stops(self.stopwords,
                             remove_puncts(self.puncts,
                                           self.word_tokenize(self.sent))))
        else:
            return wordsToStemmed(
                remove_stops(self.stopwords,
                             remove_puncts(self.puncts,
                                           self.word_tokenize(self.sent))))


class Text:
    def __init__(self, filename: str, settings: Data_settings):
        self.encoding = None
        self.settings = settings
        self.sents = list(self.fileToSents(filename))

    def fileToSents(self, filename: str) -> List[str]:
        def decode(sth: bytes, codings=None) -> str:
            if codings is None:
                codings = ["utf-8"]
            for coding in codings:
                try:
                    self.encoding = coding
                    return sth.decode(encoding=coding)
                except UnicodeDecodeError:
                    pass
            raise UnicodeDecodeError

        with open(filename, mode='rb') as file:
            text = decode(file.read()).replace('\n', ' ')
            sents = self.settings.sent_tokenize(text)
            index = 0
            for (num, sent) in enumerate(sents):
                index = text.find(sent, index)
                yield Sentence(num, re.sub("\s+", ' ', sent), index, index + len(sent), self.settings)
