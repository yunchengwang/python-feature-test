import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression


class FeatureTest:
    def __init__(self, loss='bce'):
        assert loss in ['bce', 'ce', 'mse'], f'loss not defined. Please select from ["bce", "ce", "mse"]'
        self.loss = loss
        self.sorted_features = None
        self.dim = 0

    def fit(self, X, y, n_bins, outliers=False):
        pass

    def transform(self, X, n_selected):
        assert self.sorted_features, f'Run fit() before selecting features.'
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
            partition_error = self.get_loss(y_l, y_r, self.loss)
            if partition_error < best_error:
                best_error = partition_error
                best_partition_index = i
                left_mean = left_mos.mean()
                right_mean = right_mos.mean()
        return best_error, best_partition_index, left_mean, right_mean

    @staticmethod
    def get_loss(y_l, y_r, loss):
        if loss == 'bce':
            pass
        elif loss == 'ce':
            pass
        elif loss == 'mse':
            pass
        else:
            pass
