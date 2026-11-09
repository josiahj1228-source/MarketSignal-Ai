import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="MarketSignal AI", page_icon="📈", layout="wide")
st.title("📈 MarketSignal AI")
st.caption("Historical-outcome model for product launch planning")
st.warning(
    "A probability is only as reliable as the historical data and outcome definition behind it. "
    "This app trains on data you provide; it does not fetch live competitor data."
)

with st.sidebar:
    st.header("Define success")
    target_revenue = st.number_input(
        "Revenue target within the outcome window (USD)",
        min_value=1.0, value=100000.0, step=5000.0
    )
    outcome_window = st.selectbox("Outcome window", ["12 months", "6 months", "24 months"], index=0)
    st.caption(
        "Your revenue column must represent revenue over the selected window for each historical product. "
        "If your business uses units sold, profit, or another outcome, adapt the label definition accordingly."
    )

st.subheader("1. Upload historical product outcomes")
uploaded = st.file_uploader("CSV file with one row per launched product", type=["csv"])
st.markdown(
    "Required: one numeric revenue column for the selected outcome window. "
    "Helpful predictors include launch date, category, price, marketing spend, pre-launch waitlist, "
    "review count at launch, competitor density, and customer-demand indicators."
)

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded is None:
    st.info("No file uploaded. You can download a clearly synthetic example dataset below to inspect the required structure.")
    n = 250
    rng = np.random.default_rng(7)
    launch_dates = pd.date_range("2019-01-01", periods=n, freq="15D")
    categories = rng.choice(["Home", "Beauty", "Fitness", "Pet", "Software"], n)
    price = np.round(rng.lognormal(3.2, .45, n), 2)
    marketing = np.round(rng.lognormal(8.2, .8, n), 0)
    waitlist = rng.poisson(180, n)
    reviews = rng.poisson(35, n)
    competition = rng.integers(1, 10, n)
    signal = 3.0 + .002*waitlist + .015*reviews - .12*competition - .002*price + .000025*marketing
    revenue = np.maximum(0, np.exp(signal + rng.normal(0, 1.1, n))*500)
    demo = pd.DataFrame({
        "launch_date": launch_dates.strftime("%Y-%m-%d"),
        "category": categories,
        "price_usd": price,
        "marketing_spend_usd": marketing,
        "prelaunch_waitlist": waitlist,
        "reviews_at_launch": reviews,
        "competitor_density_index": competition,
        "revenue_12m_usd": np.round(revenue, 2)
    })
    st.download_button(
        "Download synthetic example CSV",
        demo.to_csv(index=False).encode("utf-8"),
        file_name="marketsignal_synthetic_example.csv",
        mime="text/csv"
    )
    st.caption("Synthetic example only. Its outcomes are simulated and must not be used to claim real-world model performance.")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty or len(df) < 30:
    st.error("Please upload at least 30 historical product rows. In practice, several hundred or more are preferable.")
    st.stop()

st.write(f"Loaded **{len(df):,} rows** and **{len(df.columns)} columns**.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Add a numeric revenue/outcome column.")
    st.stop()

default_rev_idx = next((i for i, c in enumerate(numeric_cols) if "revenue" in c.lower() or "sales" in c.lower()), 0)
revenue_col = st.selectbox("Which column contains actual revenue for the outcome window?", numeric_cols, index=default_rev_idx)
date_candidates = [None] + [c for c in df.columns if "date" in c.lower() or "launch" in c.lower()]
date_col = st.selectbox("Launch date column (recommended for time-based holdout)", date_candidates, index=1 if len(date_candidates)>1 else 0, format_func=lambda x: "No date column" if x is None else x)

# Exclude likely identifiers and direct post-launch / outcome leakage fields from predictors.
leakage_terms = ["revenue", "sales", "profit", "units_sold", "units sold", "outcome", "success", "target", "conversion_after", "postlaunch"]
id_terms = ["id", "sku", "product_name", "product name", "url", "description"]
excluded = {revenue_col}
if date_col:
    excluded.add(date_col)
for col in df.columns:
    lower = col.lower()
    if any(term in lower for term in leakage_terms) or lower in id_terms or lower.endswith("_id"):
        excluded.add(col)
feature_candidates = [c for c in df.columns if c not in excluded]
if not feature_candidates:
    st.error("No eligible predictor columns remain. Add launch-time features such as price, category, marketing plan, or demand signals.")
    st.stop()

st.subheader("2. Choose predictors")
st.caption("Select only information that would genuinely be known at the time you want to make a prediction. Avoid post-launch data that leaks the outcome.")
features = st.multiselect("Predictor columns", feature_candidates, default=feature_candidates)
if len(features) == 0:
    st.error("Select at least one predictor.")
    st.stop()

clean = df.copy()
clean[revenue_col] = pd.to_numeric(clean[revenue_col], errors="coerce")
clean = clean.dropna(subset=[revenue_col])
clean = clean[np.isfinite(clean[revenue_col])]
clean["__success_label__"] = (clean[revenue_col] >= target_revenue).astype(int)
if clean["__success_label__"].nunique() < 2:
    st.error("The selected revenue target labels every product the same way. Choose a more informative target or provide more varied outcomes.")
    st.stop()

X = clean[features].copy()
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    else:
        X[col] = X[col].where(X[col].notna(), np.nan).astype("object")

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = [c for c in features if c not in num_features]
transformers = []
if num_features:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler())
    ]), num_features))
if cat_features:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features))
preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# Prefer a chronological holdout when a usable launch date is provided.
if date_col:
    clean["__parsed_date__"] = pd.to_datetime(clean[date_col], errors="coerce")
    dated = clean["__parsed_date__"].notna()
else:
    dated = pd.Series(False, index=clean.index)

if date_col and dated.sum() >= 30:
    ordered = clean.loc[dated].sort_values("__parsed_date__")
    split_at = int(len(ordered) * .8)
    train_idx, test_idx = ordered.index[:split_at], ordered.index[split_at:]
    split_description = "Chronological holdout: earliest 80% train, latest 20% test"
else:
    # Reproducible random split; weaker than temporal validation for launch forecasting.
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        clean.index, test_size=.2, random_state=42, stratify=clean["__success_label__"]
    )
    split_description = "Random stratified 80/20 holdout (add launch dates for a more realistic time-based test)"

y = clean["__success_label__"]
if y.loc[train_idx].nunique() < 2 or y.loc[test_idx].nunique() < 2:
    st.error("Train/test split does not contain both successful and unsuccessful products. Add data or adjust the revenue target.")
    st.stop()

model.fit(X.loc[train_idx], y.loc[train_idx])
test_prob = model.predict_proba(X.loc[test_idx])[:, 1]
test_pred = (test_prob >= .5).astype(int)
y_test = y.loc[test_idx].to_numpy()
try:
    auc = roc_auc_score(y_test, test_prob)
except ValueError:
    auc = np.nan
brier = brier_score_loss(y_test, test_prob)
acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)

st.subheader("3. Out-of-sample model check")
st.caption(split_description)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ROC AUC", "—" if np.isnan(auc) else f"{auc:.3f}")
m2.metric("Brier score ↓", f"{brier:.3f}")
m3.metric("Accuracy", f"{acc:.1%}")
m4.metric("Precision", f"{prec:.1%}")
m5.metric("Recall", f"{rec:.1%}")
st.caption("AUC measures ranking, while Brier score measures probability error (lower is better). Accuracy alone does not establish probability calibration.")
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
st.write("Confusion matrix (rows = actual, columns = predicted; threshold 50%)")
st.dataframe(pd.DataFrame(cm, index=["Actual below target", "Actual reaches target"], columns=["Predicted below target", "Predicted reaches target"]), use_container_width=True)

# Refit on all labeled historical rows for final use after holdout evaluation.
model.fit(X, y)
st.subheader("4. Score a new product")
st.caption("Fill in the same predictor fields used in historical training. The estimate refers to reaching the revenue target within the selected outcome window.")
with st.form("new_product"):
    inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(features):
        with cols[i % 2]:
            series = X[col]
            if col in num_features:
                vals = pd.to_numeric(series, errors="coerce").dropna()
                median = float(vals.median()) if len(vals) else 0.0
                minv = float(vals.min()) if len(vals) else 0.0
                maxv = float(vals.max()) if len(vals) else max(100.0, minv + 1)
                if minv == maxv:
                    maxv = minv + 1
                inputs[col] = st.number_input(col, value=median, min_value=None, max_value=None, key=f"new_{col}")
            else:
                choices = sorted([str(v) for v in series.dropna().unique()])
                choices = choices[:200]
                inputs[col] = st.selectbox(col, choices, key=f"new_{col}") if choices else st.text_input(col, key=f"new_{col}")
    submitted = st.form_submit_button("Estimate probability", type="primary")

if submitted:
    new = pd.DataFrame([{c: inputs[c] for c in features}])
    # Match numeric/categorical data types used for training.
    for col in num_features:
        new[col] = pd.to_numeric(new[col], errors="coerce")
    probability = float(model.predict_proba(new)[0, 1])
    st.markdown("### Estimated probability")
    st.progress(int(round(probability * 100)))
    st.metric(f"Chance of reaching ${target_revenue:,.0f} in {outcome_window}", f"{probability:.1%}")
    if probability >= .7:
        st.success("The model rates this product as relatively likely to reach the defined target compared with its learned historical patterns.")
    elif probability >= .4:
        st.info("The model's estimate is in an uncertain middle range. Gather stronger demand evidence and compare scenarios.")
    else:
        st.warning("The model rates this product as less likely to reach the target under its learned historical patterns.")
    st.caption("This is a model estimate, not a guarantee. Check holdout Brier score, sample size, class balance, data quality, and whether the historical launches match the new product's market.")

with st.expander("Model limitations and responsible use"):
    st.markdown("""
- The model learns associations from the uploaded historical products; it does not prove that a feature causes success.
- Historical products should represent the same market, channel, price range, and launch conditions as the product being scored.
- Features must be available at prediction time. Post-launch reviews, actual ad spend, or sales outcomes can leak future information.
- The displayed holdout metrics are not necessarily calibrated probability accuracy. Use repeated time-based validation and calibration analysis before business-critical decisions.
- Avoid using sensitive personal data. Keep product records, definitions, currencies, and outcome windows consistent.
- With small or biased samples, probabilities can be unstable. Prefer more historical launches and independent validation.
""")
