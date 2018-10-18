import re
import string
import unicodedata

import nltk

CONTENT_REGEX = re.compile(r"<p>(.*)</p>")
TAG_REGEX = re.compile(r"(<.*>)")
WHITESPACE_REGEX = re.compile(r"\s+")
TO_REPLACE = string.punctuation.replace("#","")
MATRIX = str.maketrans(TO_REPLACE + string.digits, " " * len(TO_REPLACE + string.digits))
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
ALL_LETTERS = string.ascii_letters + string.whitespace

def extract_text_content(text):
    """
    Returns text content of p tags and remove other HTML tags from string
    """
    text = " ".join(CONTENT_REGEX.findall(text))
    text = TAG_REGEX.sub("", text)
    return text

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def preprocess_string(text):
    """
    Filter string : removes tokens, stopwords, and lowercase everything
    """
    text = unicodeToAscii(text)
    text = text.lower()
    text = text.translate(MATRIX)
    text = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(text)
