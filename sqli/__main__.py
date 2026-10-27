"""python -m sqli train    # build the dataset, evaluate, save models/detector.joblib"""
import json
import sys
from pathlib import Path

from sqli.dataset import build
from sqli.model import evaluate, save_metrics


def main() -> None:
    if sys.argv[1:] != ["train"]:
        sys.exit(__doc__)
    data = build(Path("data/raw"))
    data.to_csv("data/dataset.csv", index=False)
    model, result = evaluate(data)
    Path("models").mkdir(exist_ok=True)
    model.save("models/detector.joblib")
    save_metrics(result, "data/metrics.json")
    print(json.dumps({k: v for k, v in result.items() if k != "hard_negatives"}, indent=1))
    hn = result["hard_negatives"]
    print(f"hard negatives: model {hn['model_false_positives']}/{hn['n']} false positives, "
          f"rules {hn['rules_false_positives']}/{hn['n']}; model flagged {hn['model_flagged']}")


if __name__ == "__main__":
    main()
