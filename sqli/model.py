"""The detector, a rule-based baseline, and the evaluation.

Model: TF-IDF over character n-grams (1-5, within word boundaries) and logistic
regression. Character n-grams catch the pieces of SQL syntax (`' or`, `--`, `union`,
`sleep(`) whatever the spacing, casing or encoding tricks around them; a linear model
keeps the decision explainable (each n-gram has a weight).

Baseline: the kind of keyword/character blacklist many applications still use.
"""
import json
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RULE = re.compile(r"('|--|;|/\*|\b(or|and|union|select|drop|insert|delete|update|sleep|"
                  r"benchmark|exec|waitfor)\b)", re.I)

# Benign input written by hand and never used for training: the false positives that
# matter in practice.
HARD_NEGATIVES = [
    "Siobhan O'Sullivan", "D'Arcy Fernandes", "Rock 'n' Roll Night tickets",
    "Children's books on sale", "Select a size: S, M or L", "Drop off at gate 3",
    "Order by phone or online", "Union Public Service Commission exam date",
    "Insert the SIM card, then restart", "Delete my Facebook-linked account please",
    "Update: the meeting is at 10; bring your laptop", "Where is the exam hall?",
    "I'm 5'11\" tall", "Can't find the 'Forgot password' link", "Sleep apnea treatment",
    "Benchmark scores of the RTX 3050", "Exec MBA admissions 2026",
    "Table tennis -- singles or doubles?", "between 10 and 20 kg", "Having trouble; help!",
    "\"Hello\" said O'Hara", "50% OFF -- today only", "AND gate vs OR gate truth table",
    "SQL tutorial for beginners", "How to write a SELECT statement", "union of two sets",
    "1=1 in Boolean algebra", "null pointer exception in Java", "Version 2.0; build 113",
    "Isn't it 9 o'clock already?", "user_name-2024", "pass@123", "Rs. 1,500 /- only",
    "#hashtag; @mention", "Mother's Day gifts", "St. John's Wood", "rock & roll -- the '50s",
    "Waitfor delay at platform 2", "Group by colour: red or blue", "case-sensitive search",
]


@dataclass
class Detector:
    pipeline: Pipeline

    @classmethod
    def train(cls, texts, labels) -> "Detector":
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), lowercase=True,
                                      min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
        ])
        return cls(pipe.fit(list(texts), list(labels)))

    def probability(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def predict(self, texts, threshold: float = 0.5) -> np.ndarray:
        return (self.probability(texts) >= threshold).astype(int)

    def explain(self, text: str, top: int = 6) -> list[tuple[str, float]]:
        """The character n-grams in `text` that push hardest towards "injection"."""
        tfidf, clf = self.pipeline.named_steps["tfidf"], self.pipeline.named_steps["clf"]
        row = tfidf.transform([text])
        names = tfidf.get_feature_names_out()
        contrib = [(names[j], float(v * clf.coef_[0][j])) for j, v in zip(row.indices, row.data)]
        contrib = [c for c in contrib if c[1] > 0 and c[0].strip()]
        return sorted(contrib, key=lambda c: -c[1])[:top]

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Detector":
        return joblib.load(path)


def rule_predict(texts) -> np.ndarray:
    return np.array([int(bool(RULE.search(t))) for t in texts])


def scores(y, pred) -> dict:
    return {"precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4)}


HELD_OUT_FAMILIES = ["Generic_TimeBased", "FUZZDB_Oracle", "Auth_Bypass2",
                     "payloads-sql-blind-MySQL-ORDER_BY"]


def evaluate(data: pd.DataFrame, seed: int = 7) -> tuple[Detector, dict]:
    train, test = train_test_split(data, test_size=0.2, stratify=data["label"],
                                   random_state=seed)
    model = Detector.train(train["text"], train["label"])
    pred = model.predict(test["text"])
    result = {
        "rows": {"train": len(train), "test": len(test),
                 "malicious": int(data["label"].sum()), "benign": int((1 - data["label"]).sum())},
        "test": {"model": scores(test["label"], pred),
                 "rules": scores(test["label"], rule_predict(test["text"])),
                 "confusion": confusion_matrix(test["label"], pred).tolist()},
        "hard_negatives": {
            "n": len(HARD_NEGATIVES),
            "model_false_positives": int(model.predict(HARD_NEGATIVES).sum()),
            "rules_false_positives": int(rule_predict(HARD_NEGATIVES).sum()),
            "model_flagged": [t for t, p in zip(HARD_NEGATIVES, model.predict(HARD_NEGATIVES))
                              if p]},
        "unseen_families": {},
    }
    for family in HELD_OUT_FAMILIES:            # trained without the family, tested on it
        rest, held = data[data["family"] != family], data[data["family"] == family]
        m = Detector.train(rest["text"], rest["label"])
        result["unseen_families"][family] = {
            "n": len(held), "model_recall": round(float(m.predict(held["text"]).mean()), 4),
            "rules_recall": round(float(rule_predict(held["text"]).mean()), 4)}
    final = Detector.train(data["text"], data["label"])     # the model the demo serves
    return final, result


def save_metrics(result: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
