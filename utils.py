import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelBinarizer


class FeatureTest:
    def __init__(self, loss='bce'):
        assert loss in ['bce', 'ce', 'rmse'], f'loss not supported. Please select from ["bce", "ce", "rmse"].'
        self.loss = loss
        self.dim_loss = dict()
        self.sorted_features = None
        self.dim = 0

    def fit(self, X, y, n_bins, outliers=False):
        self.dim = X.shape[1]
        for d in range(self.dim):
            min_partition_loss = self.get_min_partition_loss(X[:, d], y, n_bins, outliers)
            self.dim_loss[d] = min_partition_loss

        self.dim_loss = {k: v for k, v in sorted(self.dim_loss.items(), key=lambda item: item[1])}
        self.sorted_features = np.array(list(self.dim_loss.keys()))

    def transform(self, X, n_selected):
        assert self.sorted_features is not None, f'Run fit() before selecting features.'
        assert X.shape[1] == self.dim, f'Expect feature dimension {self.dim}, but got {X.shape[1]}.'
        return X[:, self.sorted_features[np.arange(n_selected)]]

    def fit_transform(self, X, y, n_bins, n_selected):
        self.fit(X, y, n_bins)
        return self.transform(X, n_selected)

    def get_min_partition_loss(self, f_1d, y, n_bins, outliers=False):
        if outliers:
            f_1d, y = self.remove_outliers(f_1d, y)
        min_partition_loss = float('inf')
        f_min, f_max = f_1d.min(), f_1d.max()
        bin_width = (f_max - f_min) / n_bins
        for i in range(1, n_bins):
            partition_point = f_min + i * bin_width
            y_l, y_r = y[f_1d <= partition_point], y[f_1d > partition_point]
            partition_loss = self.get_loss(y_l, y_r)
            if partition_loss < min_partition_loss:
                min_partition_loss = partition_loss
        return min_partition_loss

    def get_loss(self, y_l, y_r):
        n1, n2 = len(y_l), len(y_r)
        if self.loss == 'bce':
            lp = y_l.mean()
            if lp == 1 or lp == 0:
                lh = 0.0
            else:
                lh = np.sum(-y_l * np.log2(lp) - (1 - y_l) * np.log2(1 - lp))
            rp = y_r.mean()
            if rp == 1 or rp == 0:
                rh = 0.0
            else:
                rh = np.sum(-y_r * np.log2(rp) - (1 - y_r) * np.log2(1 - rp))
            return (lh + rh) / (n1 + n2)
        elif self.loss == 'ce':
            llb = MyLabelBinarizer()
            y_l = llb.fit_transform(y_l)
            lp = y_l.mean(axis=0)
            lh = np.sum(-y_l * np.log2(lp))
            rlb = MyLabelBinarizer()
            y_r = rlb.fit_transform(y_r)
            rp = y_r.mean(axis=0)
            rh = np.sum(-y_r * np.log2(rp))
            return (lh + rh) / (n1 + n2)
        elif self.loss == 'rmse':
            left_mse = ((y_l - y_l.mean()) ** 2).sum()
            right_mse = ((y_r - y_r.mean()) ** 2).sum()
            return np.sqrt((left_mse + right_mse) / (n1 + n2))
        else:
            pass

    @staticmethod
    def remove_outliers(f_1d, y, n_std=2.0):
        """Remove outliers for the regression problem."""
        f_mean, f_std = f_1d.mean(), f_1d.std()
        return f_1d[np.abs(f_1d - f_mean) <= n_std * f_std], y[np.abs(f_1d - f_mean) <= n_std * f_std]


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if len(self.classes_) == 1:
            return 1 - Y
        elif len(self.classes_) == 2:
            return np.hstack((Y, 1-Y))
        else:
            return Y
