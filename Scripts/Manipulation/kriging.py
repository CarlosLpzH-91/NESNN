import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Kriging:
    def __init__(self, c_v, c_v_bounds, rbf_ls, rbf_ls_bounds, n_rest, a, norm_y):
        """

        :param float c_v: Initial constant value for Kriging model.
        :param tuple c_v_bounds: Bounds of values for constant value of Kriging model (Low, High).
        :param np.ndarray rbf_ls: Initial values of Squared Exponential Kernel of Kriging Model.
        :param list rbf_ls_bounds: Bounds of values for Squared Exponential kernel of Kriging model [(Low, High),
                                                                                                     (Low, High).
                                                                                                     (Low, High)].
        :param int n_rest: Number of times the optimizer can reset.
        :param float a: Noise added at the diagonal.
        :param bool norm_y: Whether to normalize predicted values or not.
        """
        self.rbf_ls = rbf_ls
        self.rbf_ls_bounds = rbf_ls_bounds
        self.c_v = c_v
        self.c_v_bounds = c_v_bounds
        self.n_rest = n_rest
        self.a = a
        self.norm_y = norm_y

        self.kernel = C(c_v, c_v_bounds) * RBF(length_scale=self.rbf_ls,
                                               length_scale_bounds=self.rbf_ls_bounds)
        self.model = GPR(kernel=self.kernel,
                         n_restarts_optimizer=n_rest,
                         alpha=a,
                         normalize_y=norm_y)

    def train(self, samples, real_values):
        """
        Train Kriging model.

        :param np.ndarray samples: BSA parameters [[Size, Cutoff, Threshold]]
        :param np.ndarray real_values: Real aptitudes.
        :return: None.
        """
        # Re-Shape the values.
        rs_realValues = real_values.reshape(-1, 1)

        self.model.fit(samples, rs_realValues)

    def predict(self, sample):
        """
        Make predictions using Kriging model.

        :param np.ndarray or list sample: BSA parameters [[Size, Cutoff, Threshold]]
        :return: prediction: SNR predicted value [[Float]].
                 sigma: Sigma value of prediction [Float].
        """
        prediction, sigma = self.model.predict(sample, return_std=True)
        return prediction, sigma

    def report_kernel(self):
        """
        Report kernel of Kirging model.

        :return: Kernel [String]
        """
        return f'{self.model.kernel_}'

    def report_lml(self):
        """
        Report Log-Marginal Likelihood.

        :return: Log-Marginal Likelihood [Float & String].
        """
        lml = self.model.log_marginal_likelihood(self.model.kernel_.theta)
        return lml, f'{lml}'

    def report_score(self, x, y):
        """
        R2-Score.
        :param np.ndarray x: Samples
        :param np.ndarray y: Real values.
        :return: R2-Score [Float]
        """
        return self.model.score(x, y)