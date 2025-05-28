import numpy as np
import matplotlib.pyplot as plt
import config
from envs.trading_env import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle

def evaluate_model(model, env):
    """
    Evaluate a trained RL model on a given environment.

    Args:
        model: The trained RL model to evaluate.
        env: The environment to evaluate the model on.

    Returns:
        net_worths (list): Net worth values at each step.
        rewards (list): Rewards received at each step.
        actions (list): Actions taken at each step.
        dates (list): Dates corresponding to each step.
    """
    # Get initial observation
    obs_tuple = env.reset()
    # Extract observation from the tuple
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    # Run evaluation
    net_worths = []
    rewards = []
    actions = []
    dates = []  # Track actual dates for plotting
    done = False
    state = None
    while not done:
        action, state = model.predict(obs, state=state)
        step_result = env.step(action)
        obs, reward, done, info = step_result
        truncated = False
        done = done or truncated
        # Access net_worth through the underlying environment
        net_worths.append(env.envs[0].net_worth)
        rewards.append(reward)
        actions.append(action)
        # Get the current date for plotting
        dates.append(env.envs[0].get_date())
    return net_worths, rewards, actions, dates

def plot_one_instance(dates, net_worths, actions, ticker):
    """
    Plot the net worth and trading actions for a single evaluation instance.

    Args:
        dates (list): List of dates corresponding to each step.
        net_worths (list): Net worth values at each step.
        actions (list): Actions taken at each step.
        ticker (str): Ticker symbol for the asset being evaluated.
    """
    # Plot results with proper dates
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    # profit line 
    profit_ = np.array(net_worths) - np.array([config.INITIAL_BALANCE] * len(net_worths))
    plt.plot(dates, profit_, label='Profit', color='green')
    plt.plot(dates, net_worths, label='Net Worth', color='blue')
    plt.title(f'Net Worth Over Time - {ticker} (Test Data)')
    plt.ylabel('Net Worth ($)')
    plt.legend()
    plt.grid(True)

    # Format x-axis to show actual dates
    plt.gcf().autofmt_xdate()

    # Plot actions over time
    plt.subplot(2, 1, 2)

    plt.scatter(dates, [action[0][0] for action in actions], marker='o', label='Action Type')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3,label="hold threshold")  # Line at hold threshold
    plt.title('Trading Actions')
    plt.ylabel('Action Type (0-1: Buy, 1: Hold, 1-2: Sell)')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_networths_profit(net_worths,profit,ticker):
    """
    Plot net worth and profit over episodes, including summary statistics and boxplots.

    Args:
        net_worths (list or np.ndarray): Net worth values for each episode.
        profit (list or np.ndarray): Profit values for each episode.
        ticker (str): Ticker symbol for the asset being evaluated.
    """
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(net_worths, label='Net Worth', color='blue',marker='o')
    plt.axhline(y=np.mean(net_worths), color='r', linestyle='--',label=f"Avg Net worth: {np.mean(net_worths)}")  # Line at average net worth
    plt.axhline(y=config.INITIAL_BALANCE, color='black', linestyle='--',label=f"Initial Balance: {config.INITIAL_BALANCE}")  # Line at initial balance
    plt.legend()
    plt.title(f'Net Worth Over Episodes - {ticker} (Test Data)')
    plt.ylabel('Net Worth ($)')
    plt.grid(True)
    # Plot profit over time
    plt.subplot(2, 1, 2)
    plt.plot(profit, label='Profit', color='green',marker='o')
    plt.axhline(y=np.mean(profit), color='r', linestyle='--',label=f"Avg Profit: {np.mean(profit)}")  # Line at average profit
    plt.axhline(y=0, color='black', linestyle='--')  # Line at average profit
    plt.legend()
    plt.title(f'Profit Over Episodes - {ticker} (Test Data)')
    plt.ylabel('Profit ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print Maximum and minimum net worth
    print(f"Maximum Net Worth:\t{np.max(net_worths):.2f}\tMaximum Profit: {np.max(profit):.2f}")
    print(f"Minimum Net Worth:\t{np.min(net_worths):.2f}\tMinimum Profit: {np.min(profit):.2f}")

    # Boxplot for Net Worth and Profit
    data = [net_worths, profit]
    labels = ['Net Worth', 'Profit']
    colors = ['#4A90E2', '#4CAF50']

    # Setup subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)

    for i, ax in enumerate(axes):
        # Boxplot
        bp = ax.boxplot(data[i], patch_artist=True, widths=0.4, vert=True)
        
        # Style
        for box in bp['boxes']:
            box.set(facecolor=colors[i], edgecolor='black', alpha=0.7)
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set(color='black')
        for flier in bp['fliers']:
            flier.set(marker='o', color='gray', alpha=0.25, markersize=4)

        # Average annotation
        avg_val = np.mean(data[i])
        ax.text(1.05, avg_val, f"Avg {labels[i]}: {avg_val:.2f}",
                ha='left', va='center', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

        ax.set_ylabel("Value ($)", fontsize=11)
        ax.set_title(f"{labels[i]} Distribution", fontsize=12)
        ax.set_xticklabels([])
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Suptitle and layout
    plt.suptitle(f"Net Worth and Profit Over 100 Episodes - {ticker}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def load_test_model_env(MODEL, POLICY, env_path,model_path):
    """
    Load a trained RL model and its corresponding test environment.

    Args:
        MODEL (class): The RL model class (e.g., PPO, DQN).
        POLICY (str or class): The policy type to use with the model.
        env_path (str): Path to the saved environment file.
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded RL model.
        test_env: The loaded and wrapped test environment.
    """
    test_env = TradingEnv.load(path=env_path)
    test_env.set_mode('test')
    # Wrap in DummyVecEnv for compatibility witha stable-baselines
    test_env = DummyVecEnv([lambda: test_env])
    # can normalize
    # test_env = VecNormalize.load(f"../models/vecnormalize_{ticker}.pkl", test_env)
    model = MODEL(POLICY, test_env)
    model = model.load(model_path, env=test_env)
    return model, test_env