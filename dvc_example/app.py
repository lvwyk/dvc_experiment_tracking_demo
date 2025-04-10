from pathlib import Path
 
from box import ConfigBox
from dvclive import Live
from ruamel.yaml import YAML
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split
 
config = ConfigBox(YAML(typ="safe").load(Path("params.yaml").open(encoding="utf-8")))
 
 
def evaluate(model: RandomForestClassifier, X, y, split: str, live: Live):
    y_pred = model.predict(X)
    live.log_metric(f"{split}/accuracy", accuracy_score(y, y_pred))
    live.log_metric(
        f"{split}/average_precision",
        average_precision_score(y, model.predict_proba(X), average="macro"),
    )
    live.log_sklearn_plot('confusion_matrix', predictions=y_pred, labels=y)
    live.log_metric(f"{split}/recall", recall_score(y, y_pred, average="macro"))
    live.make_report()
 
X, y = datasets.load_iris(as_frame=True, return_X_y=True)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
with Live("experiments", report='md') as live:
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    evaluate(model, X_train, y_train, "train", live)
    evaluate(model, X_test, y_test, "test", live)
 