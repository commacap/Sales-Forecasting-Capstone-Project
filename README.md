# ABC-Analysis-Sales-Forecasting-Capstone-Project
# FMCG Demand Forecasting
### From Raw Sales Data to a 52-Week Forecast — Solving Stockouts & Overstocks with Prophet

---

> *A multinational FMCG company runs out of its best-selling products while simultaneously sitting on excess inventory of its worst sellers. This project uses 3 years of transactional data to understand why — and builds a forecasting system to fix it.*

---

## The Problem

FMCG supply chain teams face a paradox that costs millions annually:

| Challenge | Impact |
|-----------|--------|
| High demand volatility | Planners can't trust the numbers |
| Promotions distorting baselines | True demand is invisible behind promo spikes |
| Low forecast accuracy | Wrong orders, wrong quantities |
| Stockouts on A-products | Lost revenue on the SKUs that matter most |
| Overstocks on C-products | Capital tied up in slow-moving inventory |
| Country-level discrepancies | No consistent view across markets |
| Lack of S&OP visibility | Decisions made without a shared forecast |

This project tackles all seven — through data analysis, SKU segmentation, and machine learning forecasting.

---

##  Project Structure

```
fmcg-demand-forecasting/
│
├── project_final.ipynb          # Main notebook (Exercise 1 + Exercise 2)
├── fmcg_sales_3years_1M_rows.csv  # Source dataset (not included — see Data section)
│
├── outputs/
│   ├── abc_sku_classification.csv   # Global ABC classification results
│   ├── sku_xyz_analysis_value.csv   # Global XYZ volatility results
│   ├── prophet_forecast.csv         # 52-week forward forecast by SKU
│   └── prophet_accuracy.csv         # Backtest MAPE/MAE/RMSE per SKU
│
└── README.md
```

---

## The Dataset

| Property | Detail |
|----------|--------|
| Rows | ~1,000,000 sales transactions |
| Period | 3 years (weekly granularity) |
| Geography | 10+ countries |
| Key features | `sku_id`, `sku_name`, `country`, `units_sold`, `gross_sales`, `list_price`, `promo_flag`, `temperature`, `rain_mm`, `weekofyear`, `year` |


> **Why weekly?** FMCG replenishment cycles are weekly. Forecasting at daily level would produce false precision that the ordering process cannot act on.

---

## Exercise 1: Understanding Market Dynamics

A structured exploration of demand drivers before any modelling begins.

### a) Total Sales by Country
Revenue ranked by market — revealing which countries carry disproportionate business risk if forecast accuracy is poor.

### b) Top 20 SKUs by Revenue
A Pareto analysis confirming that a small number of SKUs drive the majority of gross sales — the foundation for ABC classification.

### c) ABC Classification (Global)
SKUs classified globally by cumulative revenue contribution:

| Class | Threshold | Revenue Share | Strategic Priority |
|-------|-----------|---------------|--------------------|
| **A** | Top 80% cumulative | ~80% | Protect — stockouts are costly |
| **B** | 80–95% cumulative | ~15% | Monitor |
| **C** | Remainder | ~5% | Rationalise — overstocks are expensive |

### d) XYZ Volatility Classification (Global)
Demand predictability measured via **Coefficient of Variation** (CV = σ / μ) on weekly gross sales:

| Class | CV Threshold | Demand Profile |
|-------|-------------|----------------|
| **X** | ≤ 0.20 | Stable, predictable |
| **Y** | ≤ 0.50 | Moderate variability |
| **Z** | > 0.50 | Volatile, erratic |

Combined ABC/XYZ segments expose the highest-risk SKUs:
- **AZ** = High value + volatile → **stockout risk**
- **CZ** = Low value + volatile → **overstock risk**

### e) Demand Drivers — Correlation Analysis
Correlation between `list_price`, `units_sold`, and `promo_flag`:
- Promotions show a consistent positive correlation with volume
- Price sensitivity varies by SKU class
- Weather (temperature, rainfall) confirmed as a seasonal demand signal

### f) Seasonality Visualisation
Weekly, monthly, and yearly weather patterns visualised against demand — confirming that Prophet's yearly seasonality component and weather regressors are justified inputs to the forecasting model.

---

##  Exercise 2: Forecasting with Prophet

### Why Prophet?

[Facebook Prophet](https://facebook.github.io/prophet/) was chosen over ARIMA or ML ensembles for three reasons:

1. **No manual tuning** — trend and seasonality are handled automatically
2. **Built-in country holiday calendars** — `add_country_holidays()` in one line
3. **Decomposable output** — the forecast is split into trend, seasonality, holidays, and regressor components, making it explainable to any business audience

### SKU Selection Strategy

10 SKUs selected from the combined ABC/XYZ segmentation to cover the full risk spectrum:

| Slots | Segment | Profile | Business Problem |
|-------|---------|---------|-----------------|
| 4 | **AY** | High-value, moderate volatility | Highest revenue frequency |
| 2 | **AZ** | High-value, volatile | Stockout risk — hardest A to plan |
| 2 | **CY** | Low-value, moderate | Quiet overstock risk |
| 2 | **CZ** | Low-value, volatile | Worst overstock offenders |

> C-class SKUs are sorted by **lowest revenue first** — deliberately targeting the most overstock-prone items.

### Model Configuration

```python
m = Prophet(
    interval_width=0.95,          # 95% confidence interval
    yearly_seasonality=True,       # learns annual patterns
    weekly_seasonality=True,       # learns within-week patterns
    daily_seasonality=False        # weekly data — no daily signal
)
```

### Step 1 — Backtesting (Model Validation)

The model is trained on all history **minus the last 13 weeks** (one quarter), then forecasts those held-out weeks. Accuracy is measured against actuals before any forward forecasting.

**Why 13 weeks?** One quarter is the standard FMCG S&OP planning horizon. A 13-week backtest directly answers: *"how well would this model have served us in last quarter's planning cycle?"*

**Metric: MAPE** (Mean Absolute Percentage Error)

| Segment | Expected MAPE | Interpretation |
|---------|--------------|----------------|
| AY | ~15–20% | Good — within FMCG benchmark |
| AZ | ~25–35% | Higher — reflects genuine volatility |
| CY | ~20–25% | Moderate — low volume adds noise |
| CZ | ~30–40% | Highest — erratic by nature |

> Higher MAPE on Z-class SKUs is **expected and informative** — it confirms that these SKUs are inherently difficult to plan, which is precisely why they cause overstocks.

### Step 2 — 52-Week Forward Forecast

After validation, the model is **refit on the complete 3-year history** and projects one full planning year ahead.

**How future weather is handled:**
No live weather API is needed. Future temperature and rainfall are estimated using **historical weekly averages per country** — the average of each week-of-year across the 3 years of data. This is the standard industry approach when external forecasts are unavailable.

**Promotions:**
Set to zero in the forward forecast (conservative baseline). To model a planned promotion, set `promo_flag = 1` for the relevant future weeks.

### Retrieving Component Plots by SKU

```python
# After running the forecast loop, view decomposition for any SKU
SKU_TO_PLOT = 'YOUR_SKU_ID'

out = sku_store[SKU_TO_PLOT]
fig = out['model'].plot_components(out['forecast'])
plt.show()
```

Each component plot shows:
- **Trend** — long-run sales direction
- **Yearly seasonality** — calendar peaks and troughs
- **Weekly seasonality** — within-week demand patterns
- **Holidays** — public holiday uplift/dip per country

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib prophet scikit-learn
```

### Run the Notebook

```bash
jupyter notebook project_final.ipynb
```

Run cells sequentially. Exercise 1 (cells 1–15) must complete before Exercise 2 (cells 16–24) — the ABC/XYZ CSV outputs from Exercise 1 are inputs to Exercise 2.

### Known Issue — Week 53

Some source data contains `weekofyear = 53` for years that only have 52 ISO weeks. This is handled automatically in the notebook:

```python
# Clips week 53 to week 52 before date construction
df['weekofyear_safe'] = df['weekofyear'].clip(upper=52)
```

---

##  Outputs

| File | Description |
|------|-------------|
| `abc_sku_classification.csv` | Every SKU with A/B/C class and cumulative revenue |
| `sku_xyz_analysis_value.csv` | Every SKU with CV and X/Y/Z volatility class |
| `prophet_forecast.csv` | 52-week forward forecast: `sku_id`, `ds`, `yhat`, `yhat_lower`, `yhat_upper` |
| `prophet_accuracy.csv` | Backtest results: MAPE, MAE, RMSE per SKU |

---

##  What's Next

This project establishes the forecasting foundation. Natural next steps:

- **Safety stock calculation** — use `yhat_upper` as the demand ceiling to derive recommended stock cover per SKU
- **Full catalogue extension** — scale from 10 to all SKUs with a single loop
- **S&OP dashboard integration** — feed `prophet_forecast.csv` into a weekly planning dashboard
- **Promotional scenario modelling** — set `promo_flag = 1` for planned weeks to quantify expected uplift
- **Quarterly model refresh** — retrain with new data each quarter to keep the forecast current

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Meta-0668E1?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

##  License

This project is for educational and portfolio purposes.


*Built as part of an FMCG Supply Chain Analytics project at Everything Data Africa. Feedback and contributions welcome.*
