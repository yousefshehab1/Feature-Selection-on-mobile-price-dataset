# Feature selection on a mobile price dataset

Three learning notebooks explore filter, wrapper and embedded feature-selection methods for predicting the `price_range` label from mobile-device attributes.

## Notebooks and data

| File | Contents |
| --- | --- |
| `feature_selection-filter.ipynb` | Variance-threshold selection and a random-forest classifier |
| `feature_selection - wrapper.ipynb` | Sequential forward selection with a decision tree and backward selection with a random forest, using mlxtend |
| `feature_selection - embedded.ipynb` | Lasso-based selection followed by a random-forest classifier |
| `train.csv` | Included data with a `price_range` target |

## Getting started

```bash
git clone https://github.com/yousefshehab1/Feature-Selection-on-mobile-price-dataset.git
cd Feature-Selection-on-mobile-price-dataset
pip install jupyter pandas numpy matplotlib seaborn scikit-learn mlxtend
jupyter notebook
```

Dependencies are listed as an installation starting point, not a tested version lock. Before running, replace the original machine-specific Windows CSV paths with the local `train.csv` path.

The wrapper notebook also reads `test.csv`, which is not included. That read is only used for inspection in the current notebook; remove that unused read and its `head()` cell or supply the original file before running all cells.

## Evaluation caveats

The saved notebook outputs are exploratory and are **not a reliable comparison of model accuracy**:

- The filter notebook applies `VarianceThreshold` to `data`, which still contains the target column, rather than to `X`. That creates target leakage when the target survives selection.
- The embedded notebook fits Lasso before the train/test split. Feature selection should be fitted on training data only, preferably inside a cross-validation pipeline. Treating ordered class labels as a regression target also needs justification.
- The three notebooks do not establish a common leakage-free evaluation protocol or a repeated-seed comparison.

Before reporting a result, separate features from the label, split the data first, fit all learned preprocessing and selection steps only on the training folds, and evaluate on untouched held-out data. No corrected benchmark or best-method claim is added in this documentation update.

## Scope

This repository is a feature-selection study, not a deployed price-prediction service. Dataset origin and reuse terms are not documented in the existing files and should be confirmed before redistributing the data elsewhere.
