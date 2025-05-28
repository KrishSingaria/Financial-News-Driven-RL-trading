import pickle
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import json
from datetime import datetime
from math import log2


from config import INITIAL_BALANCE, PRICE_DIR, SENTIMENT_DIR, TICKERS

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for trading stocks using reinforcement learning.
    ed
    This environment implements a trading simulation with the following features:
    - Continuous action space for nuanced trading decisions
    - Observation space including OHLCV data, sentiment, and account information
    - Sentiment-driven action adjustments
    - Transaction cost consideration
    - Balance stability penalties
    - Profit-based rewards with sentiment alignment
    - Support for train/test data splitting
    - Portfolio management for multiple stocks
    """
    
    def __init__(self, tickers=None, window_size=5, transaction_cost=0.001, 
                 stability_penalty_weight=0.1, sentiment_influence=0.3,
                 mode='train', train_start_date=None, train_end_date=None,
                 test_start_date=None, test_end_date=None, train_test_split=0.8):
        """
        Initialize the trading environment.
        
        Args:
            tickers: List of stock ticker symbols (defaults to TICKERS from config)
            window_size: Number of days to use for observation window
            transaction_cost: Cost per transaction as percentage
            stability_penalty_weight: Weight of stability penalty in reward
            sentiment_influence: Coefficient for sentiment influence on actions
            mode: 'train' or 'test' - determines which data subset to use
            train_start_date: Optional start date for training data (YYYY-MM-DD)
            train_end_date: Optional end date for training data (YYYY-MM-DD)
            test_start_date: Optional start date for test data (YYYY-MM-DD)
            test_end_date: Optional end date for test data (YYYY-MM-DD)
            train_test_split: Proportion of data to use for training if dates not specified
        """
        super(TradingEnv, self).__init__()
        self.tickers = tickers if tickers is not None else TICKERS
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.stability_penalty_weight = stability_penalty_weight
        self.sentiment_influence = sentiment_influence
        self.mode = mode
        
        # Dictionary to store price data for each ticker
        self.full_price_data = {}
        self.train_data = {}
        self.test_data = {}
        self.price_data = {}
        self.sentiment_data = {}
        
        # Load data for each ticker
        for ticker in self.tickers:
            # Load price data
            ticker_data = pd.read_csv(PRICE_DIR / f"{ticker}.csv")
            
            # Convert 'Date' column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(ticker_data['Date']):
                ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
            
            # Sort by date to ensure chronological order
            ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
            
            self.full_price_data[ticker] = ticker_data
            
        # Split data into train and test sets
        self._split_data(train_start_date, train_end_date, test_start_date, test_end_date, train_test_split)
        
        # Load sentiment data for each ticker
        self._load_sentiment()

        # Determine the common date range across all tickers
        self._align_data_dates()
        
        # Calculate number of steps
        first_ticker = self.tickers[0]
        self.n_steps = len(self.price_data[first_ticker]) - window_size
        
        # Initialize portfolio
        self.reset()

        # Define observation and action spaces
        # Observation for each stock: OHLCV * window + sentiment
        obs_per_stock = 5 * window_size + 1
        # Additional account info: balance + holdings and cost basis for each stock
        account_info = 1 + 2 * len(self.tickers)
        
        total_obs_size = obs_per_stock * len(self.tickers) + account_info
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # Action space: For each stock, [type, amount]
        # type: 0-1 = buy, 1 = hold, 1-2 = sell
        # amount: proportion of balance/shares to trade (0-1.0)
        self.action_space = spaces.Box(
                low=np.array([0.0, 0.0] * len(self.tickers), dtype=np.float32),
                high=np.array([2.0, 0.5] * len(self.tickers), dtype=np.float32),
                dtype=np.float32
        )
    
    def _split_data(self, train_start_date, train_end_date, test_start_date, test_end_date, train_test_split):
        """Split the data into training and testing sets based on dates or proportion."""
        for ticker in self.tickers:
            data = self.full_price_data[ticker].copy()
            
            # If specific date ranges are provided, use them
            if train_start_date and train_end_date:
                train_start = pd.to_datetime(train_start_date)
                train_end = pd.to_datetime(train_end_date)
                train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)]
                
                if test_start_date and test_end_date:
                    test_start = pd.to_datetime(test_start_date)
                    test_end = pd.to_datetime(test_end_date)
                    test_data = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)]
                else:
                    # Use data after train_end as test data if test dates not specified
                    test_data = data[data['Date'] > train_end]
            else:
                # Split based on proportion
                split_idx = int(len(data) * train_test_split)
                train_data = data.iloc[:split_idx].reset_index(drop=True)
                test_data = data.iloc[split_idx:].reset_index(drop=True)
            
            # Ensure we have enough data for both sets
            if len(train_data) <= self.window_size:
                raise ValueError(f"Training data for {ticker} has {len(train_data)} samples, need at least {self.window_size + 1}")
            
            if len(test_data) <= self.window_size:
                raise ValueError(f"Test data for {ticker} has {len(test_data)} samples, need at least {self.window_size + 1}")
            
            self.train_data[ticker] = train_data.reset_index(drop=True)
            self.test_data[ticker] = test_data.reset_index(drop=True)
            
            # Set price_data based on mode
            self.price_data[ticker] = self.train_data[ticker] if self.mode == 'train' else self.test_data[ticker]
        
        print(f"Data split complete for {len(self.tickers)} stocks")

    def _align_data_dates(self):
        """Ensure all price data has the same date range by finding common dates."""
        # Find common dates across all tickers
        if self.mode == 'train':
            date_sets = [set(self.train_data[ticker]['Date'].dt.strftime('%Y-%m-%d')) for ticker in self.tickers]
        else:
            date_sets = [set(self.test_data[ticker]['Date'].dt.strftime('%Y-%m-%d')) for ticker in self.tickers]
        
        common_dates = set.intersection(*date_sets)
        
        # Filter data to only include common dates
        for ticker in self.tickers:
            if self.mode == 'train':
                self.train_data[ticker] = self.train_data[ticker][
                    self.train_data[ticker]['Date'].dt.strftime('%Y-%m-%d').isin(common_dates)
                ].reset_index(drop=True)
                self.price_data[ticker] = self.train_data[ticker]
            else:
                self.test_data[ticker] = self.test_data[ticker][
                    self.test_data[ticker]['Date'].dt.strftime('%Y-%m-%d').isin(common_dates)
                ].reset_index(drop=True)
                self.price_data[ticker] = self.test_data[ticker]
        
        print(f"Data aligned to {len(common_dates)} common trading days")

    def _load_sentiment(self):
        """Load sentiment data for all tickers in the current price data set (train or test)."""
        self.sentiment_data = {}
        
        for ticker in self.tickers:
            dates = pd.to_datetime(self.price_data[ticker]['Date']).dt.strftime('%Y-%m-%d')
            scores = []
            for d in dates:
                f = SENTIMENT_DIR / ticker / f"{d}.json"
                if f.exists():
                    with open(f) as file:
                        scores.append(json.load(file).get("score", 0.0))
                else:
                    scores.append(0.0)
            self.sentiment_data[ticker] = scores

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = INITIAL_BALANCE
        self.initial_balance = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE  # For incremental reward
        # Initialize holdings and cost basis for each ticker
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.cost_basis = {ticker: 0 for ticker in self.tickers}
        # Calculate net worth (balance + value of all holdings)
        self.net_worth = self.balance
        # Set max steps based on the ticker with the least data
        self.max_steps = min([len(self.price_data[ticker]) for ticker in self.tickers]) - 1
        return self._get_observation(), {}
    
    def set_mode(self, mode):
        """
        Change the mode of the environment between train and test.
        
        Args:
            mode: 'train' or 'test'
        """
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'")
        
        self.mode = mode
        for ticker in self.tickers:
            self.price_data[ticker] = self.train_data[ticker] if mode == 'train' else self.test_data[ticker]
        
        self._load_sentiment()
        self._align_data_dates()
        
        # Recalculate number of steps
        first_ticker = self.tickers[0]
        self.n_steps = len(self.price_data[first_ticker]) - self.window_size
        
        self.reset()
        print(f"Environment set to {mode} mode with {len(self.price_data[first_ticker])} data points")

    def _get_observation(self):
        """Get the current observation state for all tickers (raw values, no normalization)."""
        obs = []
        # Normalization removed: using raw values for prices, volumes, balances, holdings, and cost basis
        for ticker in self.tickers:
            # OHLCV data for the window
            prices = self.price_data[ticker].iloc[self.current_step - self.window_size:self.current_step]
            ohlcv = prices[['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
            # Current sentiment
            sentiment = [self.sentiment_data[ticker][self.current_step]]
            # Combine for this ticker
            ticker_obs = np.concatenate([ohlcv, sentiment])
            obs.append(ticker_obs)
        # Flatten all ticker observations
        all_ticker_obs = np.concatenate(obs)
        # Normalize account information
        account_balance = [self.balance / self.initial_balance]
        holdings_info = []
        for ticker in self.tickers:
            holdings_info.extend([
                self.holdings[ticker],
                self.cost_basis[ticker]
            ])
        # Combine everything
        full_obs = np.concatenate([all_ticker_obs, account_balance, holdings_info])
        return np.array(full_obs, dtype=np.float32)

    def _get_price(self, ticker):
        """Get current price for a specific ticker."""
        return self.price_data[ticker].iloc[self.current_step]['Close']
    
    def get_date(self):
        """Return the current date of the environment."""
        # All tickers have the same dates after alignment
        first_ticker = self.tickers[0]
        return self.price_data[first_ticker].iloc[self.current_step]['Date']

    def _calculate_transaction_cost(self, amount):
        """Calculate transaction cost for a given trade amount."""
        return amount * self.transaction_cost

    def _calculate_stability_penalty(self):
        """Calculate penalty for balance stability."""
        balance_deviation = abs(self.net_worth - self.initial_balance) / self.initial_balance
        return -self.stability_penalty_weight * balance_deviation

    def step(self, action):
        total_transaction_cost = 0
        done = False
        truncated = False
        info = {'trades': {}, 'mode': self.mode, 'date': self.get_date()}
        prev_net_worth = self.net_worth  # For incremental reward
        # For volatility adjustment
        volatility_window = 5  # You can tune this
        alpha = 0.05  # Sentiment alignment reward coefficient (tunable)
        beta = 0.5    # Volatility scaling coefficient (tunable)
        # Process each ticker's action
        
        action = np.clip(action, self.action_space.low, self.action_space.high) ## Ensure action is within bounds
        for i, ticker in enumerate(self.tickers):
            action_type = action[i*2]
            action_amount = action[i*2 + 1]
            price = self._get_price(ticker)
            # Get current sentiment and adjust action based on it
            current_sentiment = self.sentiment_data[ticker][self.current_step]
            # Adjust action type by sentiment (as before)
            sentiment_bias = self.sentiment_influence * current_sentiment
            adjusted_action_type = np.clip(action_type + sentiment_bias, 0.0, 2.0)
            # --- (A) Adjust action amount by sentiment ---
            if adjusted_action_type < 1:  # Buy
                adjusted_action_amount = np.clip(action_amount + 0.1 * current_sentiment, 0, 0.5)
            elif adjusted_action_type > 1:  # Sell
                adjusted_action_amount = np.clip(action_amount - 0.1 * current_sentiment, 0, 0.5)
            else:
                adjusted_action_amount = action_amount
            # Initialize trade info
            info['trades'][ticker] = {
                'sentiment': current_sentiment,
                'sentiment_influence': sentiment_bias,
                'adjusted_action_type': adjusted_action_type,
                'adjusted_action_amount': adjusted_action_amount,
                'price': price
            }
            
            # Interpret action
            num_assets = len(self.tickers)
            if num_assets > 1:
                k = 1.5  # Tunable
                max_alloc_per_stock = min(1.0, k / (log2(num_assets) + 1))
            else:
                max_alloc_per_stock = 1.0
                
            if adjusted_action_type < 1:  # Buy
                max_balance_to_use = self.balance * max_alloc_per_stock
                max_shares = max_balance_to_use // price
                shares_to_buy = int(max_shares * adjusted_action_amount)
                cost = shares_to_buy * price
                if shares_to_buy > 0:
                    transaction_cost = self._calculate_transaction_cost(cost)
                    total_transaction_cost += transaction_cost
                    self.balance -= (cost + transaction_cost)
                    # Update cost basis
                    current_value = self.holdings[ticker] * self.cost_basis[ticker]
                    new_value = cost
                    total_value = current_value + new_value
                    self.holdings[ticker] += shares_to_buy
                    if self.holdings[ticker] > 0:
                        self.cost_basis[ticker] = total_value / self.holdings[ticker]
                    info['trades'][ticker]['action'] = 'buy'
                    info['trades'][ticker]['shares'] = shares_to_buy
                    info['trades'][ticker]['cost'] = cost
                    info['trades'][ticker]['transaction_cost'] = transaction_cost
                else:
                    info['trades'][ticker]['action'] = 'buy_none'
                    info['trades'][ticker]['shares'] = 0
                    info['trades'][ticker]['cost'] = 0
                    info['trades'][ticker]['transaction_cost'] = 0
                    # print(f"[DEBUG] No shares bought for {ticker} at step {self.current_step} (balance: {self.balance}, price: {price}, action_amount: {adjusted_action_amount})")
            elif adjusted_action_type > 1:  # Sell
                shares_to_sell = int(self.holdings[ticker] * adjusted_action_amount)
                if shares_to_sell > 0:
                    sale_amount = shares_to_sell * price
                    transaction_cost = self._calculate_transaction_cost(sale_amount)
                    total_transaction_cost += transaction_cost
                    self.balance += (sale_amount - transaction_cost)
                    self.holdings[ticker] -= shares_to_sell
                    info['trades'][ticker]['action'] = 'sell'
                    info['trades'][ticker]['shares'] = shares_to_sell
                    info['trades'][ticker]['revenue'] = sale_amount
                    info['trades'][ticker]['transaction_cost'] = transaction_cost
                    if self.holdings[ticker] == 0:
                        self.cost_basis[ticker] = 0
                else:
                    info['trades'][ticker]['action'] = 'sell_none'
                    info['trades'][ticker]['shares'] = 0
                    info['trades'][ticker]['revenue'] = 0
                    info['trades'][ticker]['transaction_cost'] = 0
                    # print(f"[DEBUG] No shares sold for {ticker} at step {self.current_step} (holdings: {self.holdings[ticker]}, action_amount: {adjusted_action_amount})")
            else:
                info['trades'][ticker]['action'] = 'hold'
        # Update net worth
        portfolio_value = sum(self.holdings[ticker] * self._get_price(ticker) for ticker in self.tickers)
        self.net_worth = self.balance + portfolio_value
        prev_step = self.current_step
        self.current_step += 1
        # Calculate rewards
        incremental_profit = (self.net_worth - prev_net_worth) / self.initial_balance  # Incremental reward
        stability_penalty = self._calculate_stability_penalty()
        # --- (B) Sentiment-alignment reward, (C) Volatility adjustment ---
        total_sentiment_alignment_reward = 0
        for ticker in self.tickers:
            sentiment = self.sentiment_data[ticker][prev_step]
            curr_price = self._get_price(ticker)
            prev_price = self.price_data[ticker].iloc[prev_step - 1]['Close']
            price_change = curr_price - prev_price
            # Calculate recent volatility (std of returns over window)
            start_idx = max(0, prev_step - volatility_window + 1)
            price_window = self.price_data[ticker].iloc[start_idx:prev_step + 1]['Close']
            returns = price_window.pct_change().dropna()
            volatility = returns.std() if not returns.empty else 0
            # Sentiment-alignment reward (only if sentiment and price move in same direction)
            alignment = 1 if (sentiment > 0 and price_change > 0) or (sentiment < 0 and price_change < 0) else 0
            sentiment_alignment_reward = alpha * sentiment * np.sign(price_change) * alignment
            # Volatility adjustment
            sentiment_alignment_reward *= (1 - beta * volatility)
            total_sentiment_alignment_reward += sentiment_alignment_reward
        avg_sentiment_alignment_reward = total_sentiment_alignment_reward / len(self.tickers)
        # Combine rewards
        reward = incremental_profit + stability_penalty + avg_sentiment_alignment_reward - (total_transaction_cost / self.initial_balance)
        if self.current_step >= self.max_steps:
            truncated = True
        if self.net_worth <= 0:
            done = True
        return self._get_observation(), reward, done, truncated, info

    def render(self):
        print(f"Mode: {self.mode}")
        print(f"Date: {self.get_date()}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Balance: ${self.balance:.2f}")
        
        print("\nHoldings:")
        portfolio_value = 0
        for ticker in self.tickers:
            price = self._get_price(ticker)
            value = self.holdings[ticker] * price
            portfolio_value += value
            print(f"  {ticker}: {self.holdings[ticker]} shares @ ${self.cost_basis[ticker]:.2f} (Current: ${price:.2f}, Value: ${value:.2f})")
        
        print(f"\nPortfolio Value: ${portfolio_value:.2f}")
        print(f"Net Worth: ${self.net_worth:.2f}")
        print(f"Transaction Cost: ${self._calculate_transaction_cost(self.net_worth):.2f}")
        print(f"Stability Penalty: {self._calculate_stability_penalty():.4f}")
        
        print("\nSentiment Information:")
        for ticker in self.tickers:
            sentiment = self.sentiment_data[ticker][self.current_step]
            bias = self.sentiment_influence * sentiment
            print(f"  {ticker}: {sentiment:.4f} (Bias Effect: {bias:.4f})")

    def visualize_sentiment_influence(self, n_samples=10):
        """
        Visualize how different sentiment values would affect actions.
        
        Args:
            n_samples: Number of sentiment samples to visualize
            
        Returns:
            Dictionary mapping sentiment values to their influence on action_type
        """
        sentiment_range = np.linspace(-1.0, 1.0, n_samples)
        influence = {}
        
        for sentiment in sentiment_range:
            bias = self.sentiment_influence * sentiment
            influence[round(sentiment, 2)] = {
                'bias': round(bias, 3),
                'buy_threshold': max(0, 1.0 - bias),
                'sell_threshold': min(2.0, 1.0 - bias)
            }
            
        return influence
    
    def save(self, path):
        """Save the environment state to a file."""
        # Save all attributes needed to fully restore the environment
        state = {
            # Constructor parameters
            'tickers': self.tickers,
            'window_size': self.window_size,
            'transaction_cost': self.transaction_cost,
            'stability_penalty_weight': self.stability_penalty_weight,
            'sentiment_influence': self.sentiment_influence,
            'mode': self.mode,
            
            # Current state
            'current_step': self.current_step,
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'prev_net_worth': self.prev_net_worth,
            'holdings': self.holdings,
            'cost_basis': self.cost_basis,
            'net_worth': self.net_worth,
            'max_steps': self.max_steps,
            'n_steps': self.n_steps,
            
            # Data
            'full_price_data': self.full_price_data,
            'price_data': self.price_data,
            'sentiment_data': self.sentiment_data,
            'train_data': self.train_data,
            'test_data': self.test_data
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        """Load a saved environment state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        # Create new environment instance with saved parameters
        env = cls(
            tickers=state['tickers'],
            window_size=state['window_size'],
            transaction_cost=state['transaction_cost'],
            stability_penalty_weight=state['stability_penalty_weight'],
            sentiment_influence=state['sentiment_influence'],
            mode=state['mode']
        )
        
        # Restore all saved state
        env.current_step = state['current_step']
        env.balance = state['balance']
        env.initial_balance = state['initial_balance']
        env.prev_net_worth = state['prev_net_worth']
        env.holdings = state['holdings']
        env.cost_basis = state['cost_basis']
        env.net_worth = state['net_worth']
        env.max_steps = state['max_steps']
        env.n_steps = state['n_steps']
        
        # Restore data
        env.full_price_data = state['full_price_data']
        env.price_data = state['price_data']
        env.sentiment_data = state['sentiment_data']
        env.train_data = state['train_data']
        env.test_data = state['test_data']
        
        return env
