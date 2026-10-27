"""Web demo: paste any input and see whether the model and a keyword blacklist would flag
it, and which character patterns drove the model's decision.

    python app.py            # http://localhost:5050   (run `python -m sqli train` first)
"""
import json
from pathlib import Path

from flask import Flask, render_template, request

from sqli.model import Detector, rule_predict

app = Flask(__name__)
model = Detector.load("models/detector.joblib")
metrics = json.loads(Path("data/metrics.json").read_text())
EXAMPLES = ["admin' OR '1'='1' --", "Siobhan O'Sullivan",
            "1 AND SLEEP(5)#", "Union Public Service Commission exam date",
            "' UNION SELECT username, password FROM users--", "rock 'n' roll tickets",
            "1; WAITFOR DELAY '0:0:5'--", "Can't log in to my account"]


@app.route("/")
def index():
    q = request.args.get("q", "")
    result = None
    if q:
        p = float(model.probability([q])[0])
        result = {"probability": p, "malicious": p >= 0.5, "rule": bool(rule_predict([q])[0]),
                  "evidence": model.explain(q)}
    batch = [{"text": t, "p": float(p), "rule": bool(r)} for t, p, r in
             zip(EXAMPLES, model.probability(EXAMPLES), rule_predict(EXAMPLES))]
    return render_template("index.html", q=q, result=result, metrics=metrics, batch=batch)


if __name__ == "__main__":
    app.run(port=5050)
