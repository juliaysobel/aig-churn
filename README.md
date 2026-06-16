# Customer Churn Prediction API

Predicts customer churn from RFM behavioural features. Built with FastAPI, containerized with Docker, and deployed to AWS Elastic Beanstalk.

## Project structure

```
aig-churn/
├── training/
│   └── churn_train.ipynb       # data preprocessing, model training, evaluation
├── app/
│   ├── main.py                 # FastAPI app
│   ├── requirements.txt        # dependencies
│   ├── Dockerfile              # container config
│   └── models/
│       ├── churn_model.joblib  # trained logistic regression model
│       ├── scaler.joblib       # fitted standard scaler
│       └── feature_columns.joblib
└── README.md
```

## Dataset

UCI Online Retail II (Year 2010-2011). 541k transactions, ~4,338 customers.

## Model

Two models trained and compared: Logistic Regression and Random Forest. Logistic Regression selected as the best performer (ROC-AUC: 0.7356).

A temporal split was used to avoid target leakage. Features are computed from transactions before Sep 1 2011. The churn label is derived from whether the customer made any purchase after Sep 1 2011.

## Live endpoint

```
http://aig-churn-env.eba-x5vpv2az.us-east-2.elasticbeanstalk.com/
```
```
http://aig-churn-env.eba-x5vpv2az.us-east-2.elasticbeanstalk.com/docs
```

API key: `aws-aig-churn`

## Running locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` for the prediction UI or `http://localhost:8080/docs` for the API docs.

## Running with Docker

```bash
cd app
docker build -t aig-churn .
docker run -p 8080:8080 -e API_KEY=dev-key-aig-churn aig-churn
```

## API usage

**Health check**

```
GET /health
```

**Predict churn**

```
POST /predict
X-API-Key: [your-key]
Content-Type: application/json
```

Active customer example:
```json
{
  "recency": 29,
  "frequency": 5,
  "monetary": 2790.86,
  "avg_order_value": 22.51,
  "total_items": 1590,
  "unique_products": 82
}
```

At-risk customer example:
```json
{
  "recency": 210,
  "frequency": 1,
  "monetary": 334.40,
  "avg_order_value": 19.67,
  "total_items": 197,
  "unique_products": 17
}
```

Response:
```json
{
  "churned": 0,
  "churn_probability": 0.1172,
  "model": "LogisticRegression"
}
```

## Features

| Feature | Description |
|---|---|
| recency | Days since last purchase before cutoff |
| frequency | Number of unique orders |
| monetary | Total spend (GBP) |
| avg_order_value | Mean spend per order |
| total_items | Total quantity purchased |
| unique_products | Number of distinct products bought |
