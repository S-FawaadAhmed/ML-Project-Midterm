import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

DATA_PATH = "data/sales_data.csv"
MODEL_DIR = "models"
ARTIFACT_DIR = os.path.join(MODEL_DIR, "artifacts")

NUM_COLS = ["Quantity", "Unit_Price_INR", "Discount", "Order_Month", "Order_DayOfWeek"]
CAT_COLS = ["Category", "Subcategory", "Region", "Segment", "Channel", "Payment_Method"]
ALL_FEATURES = NUM_COLS + CAT_COLS


def make_preprocessor(features, scale=True):
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in CAT_COLS]
    transformers = []
    if num:
        transformers.append(("num", StandardScaler() if scale else "passthrough", num))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(transformers=transformers)


def clean_names(names):
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def top_importance(names, values, n=15):
    order = np.argsort(values)[::-1][:n]
    return [{"feature": names[i], "importance": round(float(values[i]), 4)} for i in order]


def class_scores(pipe, X):
    """Probabilities when the model has them. Softmax of decision scores for SVM."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    d = pipe.decision_function(X)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def multiclass_metrics(y_true, preds, probs, classes):
    return {
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "macro_precision": round(precision_score(y_true, preds, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, preds, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, preds, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=classes), 4),
    }


def binary_metrics(y_true, preds, probs):
    return {
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
    }


def main():
    print("=== STARTING MODEL TRAINING PIPELINE ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek

    report = {
        "options": {c: sorted(df[c].unique().tolist()) for c in ["Category", "Region", "Segment", "Channel", "Payment_Method"]},
        "subcategories": {k: sorted(v) for k, v in df.groupby("Category")["Subcategory"].unique().items()},
        "models": {},
    }

    # ---------- Linear Regression: profit ----------
    print("\n--- Linear Regression (Profit) ---")
    X = df[ALL_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(X, df["Profit_INR"], test_size=0.2, random_state=42)
    reg = Pipeline([
        ("preprocessor", make_preprocessor(ALL_FEATURES)),
        ("rfe", RFE(estimator=LinearRegression(), n_features_to_select=45)),
        ("model", LinearRegression()),
    ])
    reg.fit(X_tr, y_tr)
    p = reg.predict(X_te)
    m = {
        "r2": round(r2_score(y_te, p), 4),
        "mae": round(mean_absolute_error(y_te, p), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, p))), 2),
    }
    print(m)
    joblib.dump(reg, os.path.join(MODEL_DIR, "profit_linear.joblib"))
    report["models"]["profit_linear"] = {
        "target": "Profit", "target_column": "Profit_INR", "model": "Linear Regression",
        "task": "regression", "features": ALL_FEATURES,
        "description": "Estimates the profit of an order in INR before fulfillment.",
        "metrics": m,
    }

    # ---------- Logistic Regression: returns ----------
    print("\n--- Logistic Regression (Return risk) ---")
    y_cls = (df["Returned"] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
    clf = Pipeline([
        ("preprocessor", make_preprocessor(ALL_FEATURES)),
        ("rfe", RFE(estimator=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                    n_features_to_select=30)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    m = binary_metrics(y_te, preds, probs)
    print(m)
    joblib.dump(clf, os.path.join(MODEL_DIR, "return_logistic.joblib"))
    report["models"]["return_logistic"] = {
        "target": "Return risk", "target_column": "Returned", "model": "Logistic Regression",
        "task": "classification", "features": ALL_FEATURES, "classes": ["No", "Yes"],
        "description": "Scores the probability that an order is returned.",
        "metrics": m, "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
    }

    # ---------- Multiclass models with their own targets ----------
    specs = {
        "channel_tree": {
            "target": "Sales channel", "target_column": "Channel", "model": "Decision Tree",
            "features": ["Category", "Subcategory", "Region", "Quantity", "Segment"],
            "description": "Predicts the sales channel from product, region, quantity and customer segment.",
            "scale": False,
            "estimator": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                                class_weight="balanced", random_state=42),
        },
        "segment_forest": {
            "target": "Customer segment", "target_column": "Segment", "model": "Random Forest",
            "features": ["Quantity", "Unit_Price_INR", "Discount", "Category", "Channel"],
            "description": "Predicts the customer segment from purchasing pattern, quantity, category and channel.",
            "scale": False,
            "estimator": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        },
        "payment_svm": {
            "target": "Payment method", "target_column": "Payment_Method", "model": "SVM",
            "features": ["Quantity", "Unit_Price_INR", "Discount", "Category", "Region", "Channel", "Segment"],
            "description": "Predicts the payment method from order characteristics, region, channel and segment.",
            "scale": True,
            "estimator": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        },
        "category_bayes": {
            "target": "Product category", "target_column": "Category", "model": "Naive Bayes",
            "features": ["Quantity", "Unit_Price_INR", "Discount", "Region", "Segment"],
            "description": "Predicts the product category from quantity, price, discount, region and segment.",
            "scale": True,
            "estimator": GaussianNB(),
        },
    }

    for key, s in specs.items():
        print(f"\n--- {s['model']} ({s['target']}) ---")
        X = df[s["features"]]
        y = df[s["target_column"]]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        pipe = Pipeline([
            ("preprocessor", make_preprocessor(s["features"], scale=s["scale"])),
            ("model", s["estimator"]),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        probs = class_scores(pipe, X_te)
        classes = list(pipe.named_steps["model"].classes_)
        m = multiclass_metrics(y_te, preds, probs, classes)
        print(m)

        entry = {
            "target": s["target"], "target_column": s["target_column"], "model": s["model"],
            "task": "classification", "features": s["features"], "classes": classes,
            "description": s["description"], "metrics": m,
            "confusion_matrix": confusion_matrix(y_te, preds, labels=classes).tolist(),
        }

        if key in ("channel_tree", "segment_forest"):
            names = clean_names(pipe.named_steps["preprocessor"].get_feature_names_out())
            est = pipe.named_steps["model"]
            entry["feature_importance"] = top_importance(names, est.feature_importances_)

            if key == "channel_tree":
                entry["rules"] = export_text(est, feature_names=names, max_depth=4)
                fig, ax = plt.subplots(figsize=(26, 11))
                plot_tree(est, feature_names=names, class_names=classes,
                          filled=True, rounded=True, fontsize=8, ax=ax)
                fig.savefig(os.path.join(ARTIFACT_DIR, "decision_tree.png"), dpi=110, bbox_inches="tight")
                plt.close(fig)
                entry["tree_image"] = "decision_tree.png"

        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{key}.joblib"))
        report["models"][key] = entry

    with open(os.path.join(ARTIFACT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nModels, report and tree image saved to models/")


if __name__ == "__main__":
    main()