## Created by Sandesh Ghimire, Sebastien Brown, Prakhar Agrawal, and Alexander Nichols
## April 10, 2024
## Amherst College Team, UChicago Algorithmic Trading Competition 2024

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

file_path = '/content/Case 2 Data 2024.csv' ## Updated this with the path to the actual file with historical trading data
data = pd.read_csv(file_path)
data.head()
data = data.iloc[:, 1:]

TRAIN = data.loc[:train_index,:].copy() # train module
TEST = data.loc[train_index+1:, :].copy() # test data


class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.window_size = 7  # Example window size for rolling statistics
        self.sharpes = []  # To store sharpe ratios for analysis

    def optimize_portfolio(self):
        cov_matrix = self.running_price_paths.cov()
        analyst_predictions = self.running_price_paths.rolling(window=self.window_size).mean() + np.random.normal(0, self.running_price_paths.std(), self.running_price_paths.shape)
        analyst_predictions = analyst_predictions.dropna()
        latest_analyst_prediction = analyst_predictions.iloc[-1]

        # Initialize weights if not already initialized
        if not hasattr(self, 'weights') or len(self.weights) != len(self.running_price_paths.columns):
            self.weights = np.array([1.0 / len(self.running_price_paths.columns)] * len(self.running_price_paths.columns))

        def objective(in_weights):
            portfolio_return = np.dot(in_weights, latest_analyst_prediction)
            portfolio_risk = np.matmul(np.matmul(in_weights, cov_matrix), np.transpose(in_weights))
            return -(portfolio_return / np.sqrt(portfolio_risk))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(-1, 1) for _ in self.running_price_paths.columns]

        result = minimize(objective, self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.sharpes.append(-result.fun)
        return result.x

    def allocate_portfolio(self, asset_prices):
        # Add the new prices to the running price paths
        new_row = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)

        # Now call optimize_portfolio to get the new weights
        weights = self.optimize_portfolio()
        return weights

def grading(train_data, test_data):
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()


## Note: We implemented yfinance while we lacked the historical trading data
