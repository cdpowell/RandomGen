"""
Created on Wed. Nov. 27th, 2024
Updated on Wed. Dec. 1st, 2024 by Christian D. Powell

v0.0a1 Changelog:
- Begins abstracting monte_carlo_rv class to allow for distributions or statistical tests to be passed to it.
- Minor cleanup.

authors: Christian D. Powell
email: cpowell74@gatech.edu
"""
__version__ = '0.0a1'
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from scipy import stats
from statsmodels.stats.stattools import durbin_watson


STATS = ['count', 'mean', 'std', 'min', '0.01', '0.05', '0.10', '0.25', '0.50', '0.75', '0.90', '0.95', '0.99', 'max']


class monte_carlo_rv():
    """Object to perform Monte Carlo simulation to obtain complete distributions and produce statistics.

    The Monte Carlo simulation works by running uniform random varibles.
    """

    def __init__(self, dist, iterations: int = 10_000, samples: int = 1, statistics: list = STATS):
        """Initialize the Monte Carlo simulation.
        """
        self.dist = dist
        self.iterations = iterations
        self.samples = samples
        self.statistics = statistics
        self.rvs = np.array([])
        self.results = list()

    def _inverse_transform_method(self):
        """Use the Inverse Transform Method to generate a random variable based off the given distribution.
        """
        return self.dist.ppf(uniform(size=1))

    def _test_statistic_sampling_method(self):
        """
        """
        # generate normal sample of data
        sample = np.random.normal(0, 1, self.samples)
        if self.dist.lower() in ['kolmogorov-smirnov', 'kolmogorov_smirnov', 'kolmogorov smirnov', 'kolmogorov', 'smirnov']:
            return stats.ks_2samp(sample, np.random.normal(size=self.samples))[0]
        elif self.dist.lower() in ['durbin-watson', 'durbin_watson', 'durbin watson', 'durbin', 'watson']:
            return stats.anderson(sample, dist='norm').statistic
        elif self.dist.lower() in ['anderson-darling', 'anderson_darling', 'anderson darling', 'anderson', 'darling']:
            return durbin_watson(sample)

    def _calculate_iteration_statistics(self):
        """Calculate the desired statistics as of the given iteration.
        """
        results = []
        # calculate the desired statistics
        if 'count' in self.statistics:
            results.append(len(self.rvs))
        if 'mean' in self.statistics:
            results.append(np.mean(self.rvs))
        if 'std' in self.statistics:
            results.append(np.std(self.rvs))
        if 'min' in self.statistics:
            results.append(np.min(self.rvs))
        for s in self.statistics:
            try:
                results.append(np.quantile(self.rvs, float(s)))
            except  Exception as e:
                pass
        if 'max' in self.statistics:
            results.append(np.max(self.rvs))
        # add the results as a numpy array
        self.results.append(np.array(results))

    def run(self, verbose=False):
        """Perform the Monte Carlo simulation.
        """
        # perform iterations
        for i in range(self.iterations):
            if verbose:
                if i % 100 == 0:
                    print(i)
            # generate random variates
            if type(self.dist) == str:
                self.rvs = np.append(self.rvs, self._test_statistic_sampling_method())
            else:
                self.rvs = np.append(self.rvs, self._inverse_transform_method())
            # calculate the iteration statistics
            self._calculate_iteration_statistics()

    def theoretical_quantiles(self, verbose: bool = False):
        """Quartile values based on using plugging the quartile float into the inverse cdf (ppf).
        """
        if verbose:
            print("Theoretical Quartile Values")
        theoretical_quartiles = list()
        quartiles = list()
        if 'min' in self.statistics:
            tmp = self.dist.ppf(0.0)
            theoretical_quartiles.append(tmp)
            quartiles.append('min')
            if verbose:
                print("min:\t\t{:.6f}".format(tmp))
        for s in self.statistics:
            try:
                tmp = self.dist.ppf(float(s))
                theoretical_quartiles.append(tmp)
                quartiles.append(s)
                if verbose:
                    print("{}%:\t\t{:.6f}".format(float(s) * 100, tmp))
            except  Exception as e:
                pass
        if 'max' in self.statistics:
            tmp = self.dist.ppf(1.0)
            theoretical_quartiles.append(tmp)
            quartiles.append('max')
            if verbose:
                print("max:\t\t{:.6f}".format(tmp))
        # return list of quartile values
        return theoretical_quartiles, quartiles

    def simulated_quantiles(self, verbose: bool = False):
        """Quartile values based on using np.quartile() on the generated random variates.
        """
        if verbose:
            print("Simulated Quartile Values")
        sim_quartiles = list()
        quartiles = list()
        if 'min' in self.statistics:
            tmp = np.min(self.rvs)
            sim_quartiles.append(tmp)
            quartiles.append('min')
            if verbose:
                print("min:\t\t{:.6f}".format(tmp))
        for s in self.statistics:
            try:
                tmp = np.quantile(self.rvs, float(s))
                sim_quartiles.append(tmp)
                quartiles.append(s)
                if verbose:
                    print("{}%:\t\t{:.6f}".format(float(s) * 100, tmp))
            except  Exception as e:
                pass
        if 'max' in self.statistics:
            tmp = np.max(self.rvs)
            sim_quartiles.append(tmp)
            quartiles.append('max')
            if verbose:
                print("min:\t\t{:.6f}".format(tmp))
        # return list of quartile values
        return sim_quartiles, quartiles

    def plot(self):
        """Display a histogram of the random variables generated in the simulation.
        """
        plt.figure()
        plt.hist(self.rvs, bins=20)
        plt.title("Distribution Histogram")
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.show()

    def plot_statistic(self, statistic: str):
        """Plot scatter plot of statistic values over the iterations
        """
        # get the index of the statistic in the list of the statistics
        i = self.statistics.index(statistic)
        if type(self.dist) != str:
            s, q = self.theoretical_quantiles()
        else:
            s, q = list(), list()
        plt.figure()
        plt.plot(list(range(self.iterations)), [self.results[j][i] for j in list(range(self.iterations))])
        # plot theoretical quartile line
        if statistic in q:
            s = s[q.index(statistic)]
            plt.axhline(s, color='r', label='theoretical value')
            plt.legend()
        plt.title(statistic)
        plt.xlabel("iteration")
        plt.ylabel("value")
        plt.show()
