# Sales Prediction Dashboard

A full-stack machine learning app built with scikit-learn, FastAPI and plain HTML. You pick a prediction target from a menu, enter order details and get a result from the matching model. A side panel shows test metrics, a confusion matrix and model-specific insights.

## Models

| Menu entry | Model | Target | Inputs |
|---|---|---|---|
| Profit | Linear Regression (RFE) | Profit_INR | All 11 order features |
| Return risk | Logistic Regression (RFE) | Returned | All 11 order features |
| Sales channel | Decision Tree | Channel | Category, Subcategory, Region, Quantity, Segment |
| Customer segment | Random Forest | Segment | Quantity, Unit_Price_INR, Discount, Category, Channel |
| Payment method | SVM (RBF kernel) | Payment_Method | Quantity, Unit_Price_INR, Discount, Category, Region, Channel, Segment |
| Product category | Naive Bayes (Gaussian) | Category | Quantity, Unit_Price_INR, Discount, Region, Segment |

Insights shown per model:
- Decision Tree: rules, tree image, feature importance, confusion matrix.
- Random Forest: feature importance, confusion matrix.
- SVM and Naive Bayes: confusion matrix and metrics.
- Linear Regression: R2, MAE and RMSE.

## Project structure

```
sales-prediction-app/
  data/
    sales_data.csv
  backend/
    __init__.py
    main.py
    schemas.py
  models/                 created by train_models.py
    artifacts/
      report.json
      decision_tree.png
  train_models.py
  index.html
  script.js
  style.css
```

## Setup

Use Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn joblib matplotlib fastapi uvicorn
```

## Run

1. Train the models from the project root. This takes about 20 seconds.

```
python train_models.py
```

2. Start the API from the project root.

```
uvicorn backend.main:app --reload
```

You also start it with `python backend\main.py`. The API runs at http://127.0.0.1:8000. Interactive docs are at http://127.0.0.1:8000/docs.

3. Open `index.html` in a browser. The badge at the top shows the backend status.

Retrain after any change to the data or to `train_models.py`, then restart the API.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Reports which models are loaded |
| GET | /models | Returns the report: metrics, features, classes, insights |
| POST | /predict/{model_key} | Returns a prediction |
| GET | /artifacts/decision_tree.png | Returns the tree image |

Model keys: `profit_linear`, `return_logistic`, `channel_tree`, `segment_forest`, `payment_svm`, `category_bayes`.

Each model requires only its own inputs. A request with a missing input returns status 422.

Example request:

```
POST /predict/channel_tree
{
  "Category": "Technology",
  "Subcategory": "Phones",
  "Region": "North",
  "Quantity": 2,
  "Segment": "Consumer"
}
```

Example response:

```
{
  "model_key": "channel_tree",
  "model_name": "Decision Tree",
  "target": "Sales channel",
  "task": "classification",
  "predicted_class": "Online",
  "probabilities": {"Marketplace": 0.2, "Online": 0.6, "Retail Store": 0.2}
}
```

## Training setup

- Split: 80/20 train and test. Classification splits are stratified. Random state is 42.
- Preprocessing: StandardScaler for numeric inputs on Linear, Logistic, SVM and Naive Bayes. Trees use raw values so the rules stay readable. One-hot encoding for categorical inputs.
- Class imbalance: Logistic Regression, Decision Tree, Random Forest and SVM use balanced class weights.
- Decision Tree: max depth 4, minimum 50 samples per leaf.
- Random Forest: 300 trees, max depth 10, minimum 5 samples per leaf.
- SVM: the API reports softmax-normalized decision scores. These show relative preference between classes and are not calibrated probabilities.

## Test results

| Model | Accuracy | F1 | ROC AUC |
|---|---|---|---|
| Linear Regression | R2 0.823 | MAE 467.79 | RMSE 746.47 |
| Logistic Regression (returns) | 0.536 | 0.134 | 0.518 |
| Decision Tree (channel) | 0.305 | 0.288 | 0.509 |
| Random Forest (segment) | 0.374 | 0.336 | 0.505 |
| SVM (payment) | 0.182 | 0.179 | 0.494 |
| Naive Bayes (category) | 0.278 | 0.129 | 0.503 |

F1 for multiclass models is the macro average. Chance accuracy is 33% for three classes and 20% for five.

## Limitations

- Profit is predictable from the inputs. Every classifier scores near chance, with ROC AUC between 0.49 and 0.52. The inputs carry almost no signal for returns, channel, segment, payment method or category.
- Treat the classification results as a finding about the data. Do not tune models to raise the numbers.
- Features known only after delivery, such as shipping days or satisfaction scores, leak information and are excluded.

## Troubleshooting

- Relative import error: start the API with `uvicorn backend.main:app --reload` from the project root, or use the updated `main.py`.
- Backend Offline badge: the API is not running, or it runs on a port other than 8000. Update `API_BASE_URL` in `script.js`.
- Could not load model report: run `train_models.py`, then restart the API.
- Dropdowns show wrong values: the dropdowns load from the trained report. Retrain after changing the data.
