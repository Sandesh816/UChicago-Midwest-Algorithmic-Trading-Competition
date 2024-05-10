## Created by Sandesh Ghimire, Sebastien Brown, Prakhar Agrawal, Alexander Nichols
## Date: April 10, 2024
## For Amherst College Team at the 2024 UChicago Algorithmic Trading Competition (Originally written in Google Collab)

#pip install xchangelib

import pandas as pd
from typing import Optional
from xchangelib import xchange_client
import asyncio
import math

# SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
# my_client = MyXchangeClient(SERVER,"amherst","charizard-snorlax-1943")
# await my_client.start()

## Focus is on individual order
class Order:
    def __init__(self, symbol, qty, price, order_type): ## order_type='Limit'
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.order_type = order_type

# book is a symbol for a stock
class Book:
    def __init__(self):
        self.bids = {} # bids is for selling
        self.asks = {} # asks is for buying
        self.maxBuyPrice = float('-inf')
        self.minSellPrice = float('inf')
        self.numOfPositions = 0
        self.latestPurchasePrice = None

    def add_order(self, price, qty, side):
        if side == 'Buy':
            if price in self.bids:
                self.bids[price] += qty
            else:
                self.bids[price] = qty
        elif side == 'Sell':
            if price in self.asks:
                self.asks[price] += qty
            else:
                self.asks[price] = qty

    def remove_order(self, price, qty, side):
        if side == 'Buy':
            if price in self.bids and self.bids[price] >= qty:
                self.bids[price] -= qty
                if self.bids[price] == 0:
                    del self.bids[price]
        elif side == 'Sell':
            if price in self.asks and self.asks[price] >= qty:
                self.asks[price] -= qty
                if self.asks[price] == 0:
                    del self.asks[price]

symbols = ['BRV', 'DLO', 'EPT', 'IGM', 'JAK', 'JMS', 'MKU', 'SCP']
price_data = pd.DataFrame(columns = symbols)
# Hope is to update the price_data if we can continuously fetch data from the Xchange server

order_books = {symbol: Book() for symbol in symbols} # Now each symbol is a book with bids and asks list

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.sum = 0
        self.symbols = ['BRV', 'DLO', 'EPT', 'IGM', 'JAK', 'JMS', 'MKU', 'SCP']
        self.order_books = {symbol: Book() for symbol in self.symbols}
        self.buy_prices = {symbol: None for symbol in self.symbols}  # Initialize buy_prices here

    def historical_mean_and_std(self):
      print("Running historical mean and std dev. Reading the file first")
      df = pd.read_csv('/content/Case1_Historical.csv', index_col=0)

      mean_prices = df.mean()
      std_dev_prices = df.std()
      max_prices = df.max()
      min_prices = df.min()
      print(f"File read complete. The mean_prices are {mean_prices}, std dev are {std_dev_prices}, max prices are {max_prices}, and min prices are {min_prices}")


      mean_EPT = df.iloc[:, 0].mean()
      mean_DLO = df.iloc[:, 1].mean()
      mean_MKU = df.iloc[:, 2].mean()
      mean_IGM = df.iloc[:, 3].mean()
      mean_BRV = df.iloc[:, 4].mean()



      # Calculate the weighted mean of the portfolio
      SCP_mean = (3 * mean_EPT + 3 * mean_IGM + 4 * mean_BRV) / (3 + 3 + 4)

      print("Weighted mean price of the SCP:", SCP_mean)

      JAK_mean = (2 * mean_EPT + 5 * mean_DLO + 3 * mean_MKU) / (3 + 3 + 4)

      print("Weighted mean price of the JAK:", JAK_mean)

      mean_prices = [mean_EPT, mean_DLO, mean_MKU, mean_IGM, mean_BRV, SCP_mean, JAK_mean]

      std_dev_EPT = df.iloc[:, 0].std()
      std_dev_DLO = df.iloc[:, 1].std()
      std_dev_MKU = df.iloc[:, 2].std()
      std_dev_IGM = df.iloc[:, 3].std()
      std_dev_BRV = df.iloc[:, 4].std()


      # Calculate the weighted standard deviation of the portfolio
      weighted_std_dev_SCP = (
          (3 * std_dev_EPT)**2 +  # Variance of column 1
          (3 * std_dev_IGM)**2 +  # Variance of column 2
          (4 * std_dev_BRV)**2    # Variance of column 3
      ) ** 0.5  # Take square root to get standard deviation

      weighted_std_dev_SCP /= 10


      print("Weighted standard deviation of the SCP:", weighted_std_dev_SCP)



      weighted_std_dev_JAK = (
          (2 * std_dev_EPT)**2 +  # Variance of column 1
          (5 * std_dev_DLO)**2 +  # Variance of column 2
          (3 * std_dev_MKU)**2    # Variance of column 3
      ) ** 0.5  # Take square root to get standard deviation

      weighted_std_dev_JAK /= 10

      print("Weighted standard deviation of the JAK:", weighted_std_dev_JAK)

      std_devs = [std_dev_EPT, std_dev_DLO, std_dev_MKU, std_dev_IGM, std_dev_BRV, weighted_std_dev_SCP, weighted_std_dev_JAK]
      relative_risks = []
      for x in std_devs:
        relative_risks.append(x/sum(std_devs))

      for y in relative_risks:
        print("Risk: ", y)

      return mean_prices, std_devs, relative_risks


    async def get_competitors_best(self):
        competitor_best = {}
        for symbol in self.symbols:
            book = self.order_books[symbol]
            best_bid = max(book.bids.keys(), default=0) if book.bids != 0 else None
            best_ask = min(book.asks.keys(), default=0) if book.asks != 0 else None
            competitor_best[symbol] = {'best_bid': best_bid, 'best_ask': best_ask}
        return competitor_best

    async def place_orders(self):
        print("Placing orders")
        competitor_best = await self.get_competitors_best()
        stop_loss_ratio = 0.98  # 5% stop-loss
        take_profit_ratio = 1.01  # 5% take-profit

        for symbol, best_prices in competitor_best.items():
            book = self.order_books[symbol]
            print(f"The current symbol is {symbol} and the current best_prices are {best_prices}")

            if best_prices['best_bid'] is None or best_prices['best_ask'] is None:
              print(f"Skipping {symbol} due to lack of complete price data.")
              continue
            ## First, checking if the stop_loss or take_profit is hit

            if book.latestPurchasePrice and book.numOfPositions != 0:
              stop_loss_price = book.latestPurchasePrice * stop_loss_ratio
              take_profit_price = book.latestPurchasePrice * take_profit_ratio

              current_bid = best_prices['best_bid']
              current_ask = best_prices['best_ask']

              ## For long positions that is num>0
              if book.numOfPositions>0:
                  if  current_bid >= take_profit_price: # current_bid <= stop_loss_price or
                    sell_quantity = min(int(book.numOfPositions * 0.75), 40)
                    await self.place_order(symbol, sell_quantity, xchange_client.Side.SELL, current_bid)
                    print(f"{'Stop-loss' if current_bid <= stop_loss_price else 'Take-profit'} hit for {symbol}, selling {sell_quantity} units at {current_bid}")
                    book.numOfPositions -= sell_quantity
                   ## For short positions that is num<0
                  if  current_ask <= take_profit_price: # current_ask >= stop_loss_price or
                    cover_quantity = min(int(abs(book.numOfPositions) * 0.75),40)
                    await self.place_order(symbol, cover_quantity, xchange_client.Side.BUY, current_ask)
                    print(f"{'Stop-loss' if current_ask >= stop_loss_price else 'Take-profit'} hit for {symbol}, covering {cover_quantity} units at {current_ask}")
                    book.numOfPositions += cover_quantity

            ## Placing buying orders on regular basis if no stop loss or no stop hit
            if best_prices['best_ask'] + 1 < book.minSellPrice:
              buy_price = best_prices['best_ask'] + 1  # Penny the competitor's ask
              buy_quantity = 10
              await self.place_order(symbol, buy_quantity, xchange_client.Side.BUY, buy_price)
              print(f"Placed buy order for {symbol} at {buy_price}")
              book.maxBuyPrice = max(buy_price, book.maxBuyPrice)
              book.numOfPositions += buy_quantity  # Update positions
            else:
              print(f"Not buying {symbol} as the best ask is not smaller than minSellPrice {book.minSellPrice}")
            ## Placing sell orders
            if best_prices['best_bid'] - 1 > book.maxBuyPrice:
              sell_price = best_prices['best_bid'] - 1
              sell_quantity = 10
              await self.place_order(symbol, sell_quantity, xchange_client.Side.SELL, sell_price)
              print(f"Placed sell order for {symbol} at {sell_price}")
              book.minSellPrice = min(sell_price, book.minSellPrice)
              book.numOfPositions -= sell_quantity  # Update positions
            else:
              print(f"Not selling {symbol} as the best bid is not higher than maxBuyPrice {book.maxBuyPrice}")

            # await asyncio.sleep(5)  # Short delay before next iteration


    async def trade(self):
        fair_prices = []
        std_deviations = []
        risks = []
        #fair_prices, std_deviations, risks = self.historical_mean_and_std()
        while True:
            try:
                await self.place_orders()
                await asyncio.sleep(5)  # Pause between trade cycles
            except asyncio.CancelledError:
                print("Trading task was cancelled.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                await asyncio.sleep(5)  # Wait before retrying

            print("My positions:", self.positions)


    async def bot_handle_book_update(self, symbol: str): ## , bids:dict, asks:dict
          book = self.order_books[symbol]
          # book.bids = bids
          # book.asks = asks


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        '''
        Idea: First check if the order exists in our open orders. If the cancel was successful then delete the order (recognized from the order id) from the open_orders
        '''
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled") ## So, order[1]= quantity and order[2] = type

        if success:
            del self.open_orders[order_id]
        else:
            print(f"Error cancelling order {order_id}: {error}")

        # Print the current state of open orders
        print("Current open orders:")

        for id, ord in self.open_orders.items():
            print(f"Order ID: {id}, Symbol: {ord.symbol}, Qty: {ord.qty}, Remaining: {ord.remaining_qty}, Price: {ord.price}, Side: {ord.side}, Type: {ord.order_type}")


    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        # if order_id in self.open_orders:
        #     order = self.open_orders[order_id] ## Get the order

        # if order.side == 'Buy':
        #     self.positions[order.symbol] = self.positions.get(order.symbol, 0) + qty
        # elif order.side == 'Sell':
        #     self.positions[order.symbol] = self.positions.get(order.symbol, 0) - qty

        print("order fill", self.positions)


    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass



    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass
    # ## Get the books of all the orders atm of all the stocks
    # def getBooks():
    #   for i in symbols:
    #     book = self.order



    async def view_books(self):
        """Prints the books every 3 seconds."""
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        asyncio.create_task(self.trade())
        # asyncio.create_task(self.view_books())
        await self.connect()


async def main():
    SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER,"amherst","charizard-snorlax-1943")
    await my_client.start()
    return

# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     result = loop.run_until_complete(main())
await main()

