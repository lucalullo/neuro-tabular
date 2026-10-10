"""Mixed-data DataFrame pipeline, CV and a small usability grid-search smoke."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neurotabular import NeuroTabularClassifier


def main():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            "amount": rng.normal(size=96),
            "city": pd.Categorical(np.resize(["Rome", "Milan", None], 96)),
            "active": pd.Series(np.resize([True, False, None], 96), dtype="boolean"),
        }
    )
    X.loc[::13, "amount"] = np.nan
    y = np.arange(len(X)) % 2
    columns = ColumnTransformer(
        [("scale", StandardScaler(), ["amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    pipeline = Pipeline(
        [
            ("columns", columns),
            (
                "model",
                NeuroTabularClassifier(max_epochs=2, device="cpu", random_state=7),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
    print("CV accuracy:", cross_val_score(pipeline, X, y, cv=cv))
    search = GridSearchCV(
        pipeline, {"model__hidden_dim": [8, 12]}, cv=cv, error_score="raise"
    ).fit(X, y)
    model = search.best_estimator_.named_steps["model"]
    if hasattr(model, "input_schema_"):
        print("Schema:", pd.DataFrame(model.input_schema_))
    else:
        # Public 0.3.1 predates this optional development diagnostic.
        print("Features:", model.feature_names_in_)
    print("Device:", model.device_)
    print("History:", pd.DataFrame(model.history_))
    print("Predictions:", search.predict(X.iloc[:5]))
    # These small epochs and grid demonstrate API use, not predictive quality.


if __name__ == "__main__":
    main()
