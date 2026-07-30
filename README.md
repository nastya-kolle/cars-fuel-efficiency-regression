# Auto MPG Regression — Predicting Fuel Efficiency

Regression analysis of the classic Auto MPG dataset (car models collected at a US university in 1983): data cleaning, exploratory analysis, and comparison of multiple regression approaches — linear, non-linear (linearized via log transform), polynomial, and decision tree — to predict a car's fuel efficiency (`mpg`).

## About the project

**Goal:** build at least two regression models (linear and non-linear, the latter linearized via a log transformation) explaining the relationship between `mpg` and other features — `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, `origin`, `car name` — compare their quality, and use the best one to make a prediction.

**What was done:**
- Data cleaning: fixing the `horsepower` column (non-numeric values, missing data imputed by group means based on `origin` and `model year`)
- Distribution analysis of quantitative features (histograms, pairplot, Shapiro–Wilk normality test)
- Analysis of categorical features' influence on `mpg` (boxplots by `origin`, `model year`, `cylinders`; Kruskal–Wallis test)
- Correlation analysis between quantitative features (Pearson correlation, heatmap)
- Linear and log-linearized (power-law) regression models with `statsmodels` (OLS), including multi-factor models
- Train/test split and comparison of five model families: **Linear Regression**, **Non-linear (log-linearized) Regression**, **Polynomial Regression**, and **Decision Tree Regression** — each with a reduced and a full feature set
- Model comparison by R² (train/test) and MSE (train/test), plus a sample forecast for a new car
- Final conclusion on the best-performing model

**Key finding:** the log-linearized non-linear regression and the full polynomial/decision tree models showed the best R² and MSE on both train and test sets, with the non-linear (log-linearized) model recommended for further use.

## Example visualizations

**Distribution of fuel efficiency (mpg)**

![MPG histogram](mpg_histogram.png)

**Pairwise relationships between quantitative features**

![Pairplot of quantitative features](pairplot_quantative.png)

**Fuel efficiency by car origin**

![Boxplot origin vs mpg](boxplot_origin_mpg.png)

**Correlation heatmap of quantitative features**

![Correlation heatmap](correlation_heatmap.png)

**Weight vs mpg — linear fit**

![Weight vs mpg jointplot](weight_mpg_jointplot.png)

**Log(displacement) vs log(mpg) — linearized power-law fit**

![Log displacement vs log mpg jointplot](log_displacement_mpg_jointplot.png)

**Decision tree (simplified)**

![Decision tree simple](decision_tree_simple.png)

**Decision tree (full feature set)**

![Decision tree full](decision_tree_full.png)

## Repository structure

```
├── Cars_Regression.ipynb   # main analysis notebook
├── *.png                   # charts used in the README
└── README.md
```

## Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` · `statsmodels` · `scikit-learn`

## How to run

1. Clone the repository
2. Place the `auto-mpg.csv` dataset in the project folder
3. Open `Cars_Regression.ipynb` in Jupyter Notebook / Google Colab
4. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn`
5. Run the cells in order
