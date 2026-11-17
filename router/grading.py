import re
import unicodedata

ANSWER = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def final_part(text):
    found = ANSWER.findall(text)
    if found:
        return found[-1].strip()
    return text.strip()


def to_number(s):
    s = s.replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    return float(nums[-1])


def grade_math(text, gold):
    got = to_number(final_part(text))
    if got is None:
        return False
    try:
        return abs(got - float(gold[0])) < 1e-6
    except ValueError:
        return False


def grade_mc(text, gold):
    found = ANSWER.findall(text)
    if found:
        letters = re.findall(r"\b([A-E])\b", found[-1])
        return bool(letters) and letters[0] == gold[0]
    letters = re.findall(r"\b([A-E])\b", text)
    return bool(letters) and letters[-1] == gold[0]


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def grade_fact(text, gold):
    got = " " + norm(final_part(text)) + " "
    for g in gold:
        g = norm(str(g))
        if g and (" " + g + " ") in got:
            return True
    return False


def grade(kind, text, gold):
    if kind == "math":
        return grade_math(text, gold)
    if kind == "mc":
        return grade_mc(text, gold)
    return grade_fact(text, gold)
