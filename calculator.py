

from typing import Tuple

import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.stats import gaussian_kde, kurtosis, skew


class Calculator:

    def __init__(self, kde_dots: int = 1000, fft_dots: int = 4096):
        self.kde_dots = kde_dots
        self.fft_dots = fft_dots
        self._coordinates = None

    def __len__(self):
        return len(self.coordinates)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        """3D coordinates.

        Parameters
        ----------
        value : np.ndarray
            Numpy 3D array of dynamic system coordinates.

        """
        self._coordinates = value

    def check_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate minimum and maximum for X, Y, Z coordinates.
        """
        return np.min(self.coordinates, axis=0), np.max(self.coordinates, axis=0)

    def check_moments(self, is_common: bool = False) -> dict:
        """Calculate stochastic parameters: mean, variance, skewness, kurtosis etc.

        Parameters
        ----------
        is_common : bool
            If False - method returns moments for each coordinate. Otherwise
            returns moments over all ndarray. Similar for axis or axes along
            which the moments are computed.
            The default is to compute the moments for each coordinate.
        """
        axis = None if is_common else 0
        return {
            "Mean": np.mean(self.coordinates, axis=axis),
            "Variance": np.var(self.coordinates, axis=axis),
            "Skewness": skew(self.coordinates, axis=axis),
            "Kurtosis": kurtosis(self.coordinates, axis=axis),
            "Median": np.median(self.coordinates, axis=axis),
        }

    def check_probability(self):
        """Check probability for each chaotic coordinates.

        """
        p_axi = np.zeros([3, self.kde_dots])
        d_kde = np.zeros([3, self.kde_dots])
        for ii in range(3):
            p_axi[ii] = np.linspace(self.coordinates[ii, :].min(), self.coordinates[ii, :].max(), self.kde_dots)
            d_kde[ii] = gaussian_kde(self.coordinates[ii, :]).evaluate(p_axi[ii, :])
            d_kde[ii] /= d_kde[ii].max()
        return d_kde

    def calculate_spectrum(self):
        """Calculate FFT (in dB) for input 3D coordinates. You can set number of FFT points into the object instance.

        """
        spectrum = fft(self.coordinates, self.fft_dots, axis=0)
        spectrum = np.abs(fftshift(spectrum, axes=0))
        # spectrum = np.abs(spectrum)
        spectrum /= np.max(spectrum)
        spec_log = 20 * np.log10(spectrum + np.finfo(np.float32).eps)
        return spec_log

    def calculate_correlation(self):
        """Calculate auto correlation function for chaotic coordinates.

        """
        nn, mm = 3, len(self.coordinates)
        auto_corr = np.zeros([mm, nn])
        for ii in range(nn):
            auto_corr[:, ii] = np.correlate(self.coordinates[:, ii], self.coordinates[:, ii], "same")
        return auto_corr


if __name__ == "__main__":
    calc = Calculator()

    num_dots = 10000
    np.random.seed(42)
    calc.coordinates = np.random.randn(num_dots, 3)

    dd_kde = calc.check_probability()
    import matplotlib.pyplot as plt

    print(dd_kde.shape)
    plt.figure("Probability density function")
    for idx in range(dd_kde.shape[0]):
        plt.plot(dd_kde[idx], ".")
        # plt.xlim([0, dd_kde.shape[1] - 1])
        plt.grid()
    plt.tight_layout()
    plt.show()
