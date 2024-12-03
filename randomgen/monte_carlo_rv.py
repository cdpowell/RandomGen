"""
Created on Wed. Nov. 27th, 2024

authors: Christian D. Powell
email: cpowell74@gatech.edu
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform


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
            self.rvs = np.append(self.rvs, self.dist.ppf(uniform(size=1)))
            # calculate the iteration statistics
            self._calculate_iteration_statistics()

    def theoretical_quartiles(self, verbose: bool = False):
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

    def simulated_quartiles(self, verbose: bool = False):
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
        s, q = self.theoretical_quartiles()
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
