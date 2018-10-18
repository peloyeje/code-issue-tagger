import re
import string

import nltk

CONTENT_REGEX = re.compile(r"<p>(.*)</p>")
TAG_REGEX = re.compile(r"(<.*>)")
WHITESPACE_REGEX = re.compile(r"\s+")
MATRIX = str.maketrans(string.punctuation + string.digits, " " * len(string.punctuation + string.digits))
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def extract_text_content(text):
    """
    Returns text content of p tags and remove other HTML tags from string
    """
    text = " ".join(CONTENT_REGEX.findall(text))
    text = TAG_REGEX.sub("", text)
    return text

def preprocess_string(text):
    """
    Filter string : removes tokens, stopwords, and lowercase everything
    """
    text = text.lower()
    text = text.translate(MATRIX)
    text = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(text)
