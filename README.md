# python-feature-test

Python implementation for the Discriminant Feature Test (DFT) 
and Relevant Feature Test (RFT) in sklearn style.

## Requirements

    numpy==1.21.5
    scikit-learn==1.0.1

## Binary Classification

Adopt the binary cross-entropy (BCE) loss for feature selection.
Implementation of the loss function follows the equation 
[here](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html).

    dft = FeatureTest(loss='bce')

## Regression

Adopt the root mean-squared-error (RMSE) as the loss function for
feature selection. Consider turning on `outliers` in `fit()` if there
are some anomalies in the datasets.
Implementation of the loss function follows the equation 
[here](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html).

    rft = FeatureTest(loss='rmse')

## Multi-class Classification

Use the cross-entropy (CE) loss for multi-class classification `(n_class > 2)` 
problems.

    dft = FeatureTest(loss='ce')

## Multi-label Classification

Multi-label classification can be formulated as `n_class` independent
binary classification. Consider using BCE loss to select features
for the multi-label classification problems.

## Demo

The basic usage of the package is included in the [demo](./demo.ipynb).
More demos will be available soon.

## Citation

Please consider citing the original feature test paper if you find 
this code useful.

```
@article{yang2022supervised,
  title={On supervised feature selection from high dimensional feature spaces},
  author={Yang, Yijing and Wang, Wei and Fu, Hongyu and Kuo, C-C Jay and others},
  journal={APSIPA Transactions on Signal and Information Processing},
  volume={11},
  number={1},
  year={2022},
  publisher={Now Publishers, Inc.}
}
```