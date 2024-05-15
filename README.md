# 2024 UChicago Midwest Algorithmic Trading Competition 

This repository contains the code for the trading strategies developed for the 2024 UChicago Midwest Algorithmic Trading Competition by Amherst College team,
comprising Sandesh Ghimire, Sebastien Brown, Prakhar Agrawal, and Alexander Nichols

The project involved two cases: Case 1: Building a live trading bot and Case 2: Building a portfolio optimization model

## Case 1: Live Trading Bot

The live trading bot is designed to optimize trading strategies based on real-time market data. It uses historical data to calculate weighted mean prices 
and standard deviations for risk assessment.

### Key Features

- **Historical Data Analysis**: Calculates weighted mean prices and standard deviations
- **Order Placement**: Optimizes buy and sell orders based on stop-loss and take-profit conditions
- **Real-time Updates**: Handles order book updates and market responses


## Case 2: Portfolio Optimization Model
The portfolio optimization model uses historical stock data to balance risk and return over a ten-year period. It dynamically adjusts portfolio weights to 
maximize the Sharpe ratio.

### Key Features
- **Risk Assessment**: Analyzes historical stock data to assess risk
- **Portfolio Allocation**: Optimizes portfolio weights using rolling windows and analyst predictions
- **Performance Evaluation**: Simulates portfolio performance and visualizes capital growth
