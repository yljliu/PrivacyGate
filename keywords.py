import re


TASK_CUES = re.compile(
    r"\b(e-?mail|dear|sincerely|doctor|symptom|diagnos\w*|medication|resume|cover letter|memo|contract)\b",
    re.I,
)
DISCLOSURE_CUES = re.compile(
    r"\b(my (husband|wife|boss|manager|son|daughter|doctor)|i was diagnosed|password|salary|loan|credit card|my name is)\b",
    re.I,
)
IDENTIFIER_PATTERNS = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
)


def keyword_match(text):
    return bool(TASK_CUES.search(text)) and bool(
        DISCLOSURE_CUES.search(text) or IDENTIFIER_PATTERNS.search(text)
    )