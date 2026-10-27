import pandas as pd
import pytest

from sqli.model import HARD_NEGATIVES, Detector, rule_predict

ATTACKS = ["' OR 1=1--", "admin'--", "1 UNION SELECT null,version()--", "1 AND SLEEP(5)",
           "'; DROP TABLE users;--", "1' ORDER BY 3#", "' or ''='", "1;waitfor delay '0:0:5'--"]
BENIGN = ["John Smith", "laptop bag", "Where is my order?", "priya@example.com", "42",
          "It was the best of times", "She said yes", "Mumbai, India", "blue jeans size 32",
          "Thanks for your help!"]


@pytest.fixture(scope="module")
def model():
    data = pd.DataFrame({"text": ATTACKS * 5 + BENIGN * 5,
                         "label": [1] * len(ATTACKS) * 5 + [0] * len(BENIGN) * 5})
    return Detector.train(data["text"], data["label"])


def test_the_model_separates_attacks_from_ordinary_input(model):
    assert model.predict(["' OR '1'='1' --", "x' UNION SELECT password FROM users--"]).all()
    assert not model.predict(["Jane Doe", "Where is the shop?"]).any()


def test_explanations_point_at_sql_syntax(model):
    evidence = model.explain("' OR 1=1--")
    assert evidence and all(w > 0 for _, w in evidence)


def test_the_blacklist_flags_apostrophes_in_names():
    assert rule_predict(["O'Brien"])[0] == 1          # the false positive the model avoids
    assert rule_predict(["hello world"])[0] == 0
    assert len(HARD_NEGATIVES) == 40
