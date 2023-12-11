import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import datetime
import io
import base64

# ---------------------- Data Fetching and Processing ----------------------

def fetch_historical_data(symbol):
    today = datetime.date.today()
    next_day = today + datetime.timedelta(days=2)
    data = yf.download(symbol, start="2022-06-01", end=next_day.strftime('%Y-%m-%d'))
    return data

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(data, window, deviation):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['bb_middle'] = rolling_mean
    data['bb_upper'] = rolling_mean + (deviation * rolling_std)
    data['bb_lower'] = rolling_mean - (deviation * rolling_std)
    return data

def identify_support_resistance(data, order=5, support_count=4, resistance_count=4):
    data['min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order=order)[0]]['Close']
    data['max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order=order)[0]]['Close']
    support_levels = data['min'].value_counts().head(support_count)
    resistance_levels = data['max'].value_counts().head(resistance_count)
    return support_levels, resistance_levels

def identify_double_tops_bottoms(data, tolerance=0.05):
    local_max = data['max'].dropna()
    local_min = data['min'].dropna()
    double_tops = local_max[(local_max.shift(1) / local_max - 1).abs() < tolerance]
    double_bottoms = local_min[(local_min.shift(1) / local_min - 1).abs() < tolerance]
    return double_tops, double_bottoms

def find_cross_points(data, fast_column, slow_column):
    cross_up = ((data[fast_column].shift(1) < data[slow_column].shift(1)) & (data[fast_column] > data[slow_column]))
    cross_down = ((data[fast_column].shift(1) > data[slow_column].shift(1)) & (data[fast_column] < data[slow_column]))
    return cross_up, cross_down

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------- Visualization Functions ----------------------

def visualize_basic(data, symbol, double_tops, double_bottoms, support_levels, resistance_levels):
    # Extend the data by 5 days for the prediction
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    extended_data = data.reindex(data.index.union(date_range))

    plt.style.use('dark_background')
    plt.figure(figsize=(16, 10))

    # Plotting Close Price
    plt.plot(extended_data['Close'], color='#1E90FF', alpha=1, label='Close Price', linewidth=1.5)

    # Plotting Support and Resistance levels
    for support in support_levels.index:
        plt.axhline(y=support, color='green', linestyle='--')
        plt.text(extended_data.index[-1], support, 'Support', fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='green')

    for resistance in resistance_levels.index:
        plt.axhline(y=resistance, color='red', linestyle='--')
        plt.text(extended_data.index[-1], resistance, 'Resistance', fontsize=8, verticalalignment='top', horizontalalignment='right', color='red')

    # Annotate the last double top or double bottom
    if not double_tops.empty:
        last_double_top_date = double_tops.index[-1]
        last_double_top_value = double_tops.values[-1]
        plt.scatter(last_double_top_date, last_double_top_value, color='#FF1493', marker='o', s=80, edgecolors='black')

        # Draw the dashed line to the next support level after identifying a double top
        closest_support = support_levels.index[support_levels.index < double_tops[-1]].max()
        if not np.isnan(closest_support):
            plt.plot([last_double_top_date, extended_data.index[-1]], [last_double_top_value, closest_support], 'y--')
            plt.annotate('Buy range', (extended_data.index[-1], closest_support), textcoords="offset points", xytext=(-10,-15), ha='center', color='white')

    if not double_bottoms.empty:
        last_double_bottom_date = double_bottoms.index[-1]
        last_double_bottom_value = double_bottoms.values[-1]
        plt.scatter(last_double_bottom_date, last_double_bottom_value, color='#00BFFF', marker='o', s=80, edgecolors='black')

        # Draw the dashed line to the next resistance level after identifying a double bottom
        closest_resistance = resistance_levels.index[resistance_levels.index > double_bottoms[-1]].min()
        if not np.isnan(closest_resistance):
            plt.plot([last_double_bottom_date, extended_data.index[-1]], [last_double_bottom_value, closest_resistance], 'y--')
            plt.annotate('Sell range', (extended_data.index[-1], closest_resistance), textcoords="offset points", xytext=(-10,10), ha='center', color='white')

    plt.title(f'Technical Analysis of {symbol}', fontsize=15, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price', fontsize=12, color='white')
    plt.grid(color='#2C2C2C', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.savefig(img_data, format='png')
    img_data = io.BytesIO()
    plt.close()
    return base64.b64encode(img_data.getvalue()).decode()

def visualize_adv(data, symbol, double_tops, double_bottoms, support_levels, resistance_levels):
    # Extend the data by 5 days for the prediction
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    extended_data = data.reindex(data.index.union(date_range))

    plt.style.use('dark_background')
    plt.figure(figsize=(16, 10))

    # Plotting Close Price
    plt.plot(extended_data['Close'], color='#1E90FF', alpha=1, label='Close Price', linewidth=1.5)

    # Plotting Support and Resistance levels
    for support in support_levels.index:
        plt.axhline(y=support, color='green', linestyle='--')
        plt.text(extended_data.index[-1], support, 'Support', fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='green')

    for resistance in resistance_levels.index:
        plt.axhline(y=resistance, color='red', linestyle='--')
        plt.text(extended_data.index[-1], resistance, 'Resistance', fontsize=8, verticalalignment='top', horizontalalignment='right', color='red')

    # Plotting double tops and bottoms with distinct colors and larger markers
    plt.scatter(double_tops.index, double_tops, color='#FF1493', marker='o', label='Double Top', s=80, edgecolors='black')
    plt.scatter(double_bottoms.index, double_bottoms, color='#00BFFF', marker='o', label='Double Bottom', s=80, edgecolors='black')

    # Annotate the last double top or double bottom
    if not double_tops.empty:
        last_double_top_date = double_tops.index[-1]
        last_double_top_value = double_tops.values[-1]
        plt.scatter(last_double_top_date, last_double_top_value, color='#FF1493', marker='o', s=80, edgecolors='black')

        # Draw the dashed line to the next support level after identifying a double top
        closest_support = support_levels.index[support_levels.index < double_tops[-1]].max()
        if not np.isnan(closest_support):
            plt.plot([last_double_top_date, extended_data.index[-1]], [last_double_top_value, closest_support], 'y--')
            plt.annotate('Buy range', (extended_data.index[-1], closest_support), textcoords="offset points", xytext=(-10,-15), ha='center', color='white')

    if not double_bottoms.empty:
        last_double_bottom_date = double_bottoms.index[-1]
        last_double_bottom_value = double_bottoms.values[-1]
        plt.scatter(last_double_bottom_date, last_double_bottom_value, color='#00BFFF', marker='o', s=80, edgecolors='black')

    # Plotting the predictive dashed line based on the last double top or double bottom signal
    if not double_tops.empty and (double_bottoms.empty or double_tops.index[-1] > double_bottoms.index[-1]):
        closest_support = support_levels.index[support_levels.index < double_tops[-1]].max()
        if not np.isnan(closest_support):
            plt.plot([double_tops.index[-1], extended_data.index[-1]], [double_tops[-1], closest_support], 'y--', label='Predicted Trend')
    elif not double_bottoms.empty:
        closest_resistance = resistance_levels.index[resistance_levels.index > double_bottoms[-1]].min()
        if not np.isnan(closest_resistance):
            plt.plot([double_bottoms.index[-1], extended_data.index[-1]], [double_bottoms[-1], closest_resistance], 'y--', label='Predicted Trend')

    plt.title(f'Technical Analysis of {symbol}', fontsize=15, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price', fontsize=12, color='white')
    plt.grid(color='#2C2C2C', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.savefig(img_data, format='png')
    img_data = io.BytesIO()
    plt.close()
    return base64.b64encode(img_data.getvalue()).decode()

def visualize_pro(data, support_levels, resistance_levels, cross_up, cross_down, double_tops, double_bottoms, symbol):
    # Extend the data by 5 days for the prediction
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
    extended_data = data.reindex(data.index.union(date_range))
    
    # Extend boolean series to match the length of the extended data
    cross_up = cross_up.reindex(extended_data.index, fill_value=False)
    cross_down = cross_down.reindex(extended_data.index, fill_value=False)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(16,10))
    
    # Customizing color and style for support and resistance levels
    for level in support_levels.index:
        plt.axhline(level, color='#00FF00', linestyle='dashed', alpha=0.7, linewidth=0.8)
        plt.text(extended_data.index[-1], level, f"Support {level:.2f}", color='#00FF00', ha='left', va='bottom', fontsize=9)

    for level in resistance_levels.index:
        plt.axhline(level, color='#FF4500', linestyle='dashed', alpha=0.7, linewidth=0.8)
        plt.text(extended_data.index[-1], level, f"Resistance {level:.2f}", color='#FF4500', ha='left', va='top', fontsize=9)

    # Plotting price and Bollinger Bands with customized colors
    plt.plot(extended_data['Close'], color='#1E90FF', alpha=1, label='Close Price', linewidth=1.5)
    plt.plot(extended_data['bb_upper'], color='#FFD700', linestyle='--', label='Upper Bollinger Band', linewidth=0.9)
    plt.plot(extended_data['bb_middle'], color='#00FA9A', linestyle='--', label='Middle Bollinger Band', linewidth=0.9)
    plt.plot(extended_data['bb_lower'], color='#FFD700', linestyle='--', label='Lower Bollinger Band', linewidth=0.9)

    # Plotting EMA cross points with larger, more visible markers
    plt.scatter(cross_up[cross_up].index, extended_data['Close'][cross_up], color='#ADFF2F', marker='^', label='EMA Cross Up', s=60)
    plt.scatter(cross_down[cross_down].index, extended_data['Close'][cross_down], color='#FF6347', marker='v', label='EMA Cross Down', s=60)

    # Plotting double tops and bottoms with distinct colors and larger markers
    plt.scatter(double_tops.index, double_tops, color='#FF1493', marker='o', label='Double Top', s=80, edgecolors='black')
    plt.scatter(double_bottoms.index, double_bottoms, color='#00BFFF', marker='o', label='Double Bottom', s=80, edgecolors='black')

    # Plotting the predictive dashed line based on the last double top or double bottom signal
    if not double_tops.empty and (double_bottoms.empty or double_tops.index[-1] > double_bottoms.index[-1]):
        closest_support = support_levels.index[support_levels.index < double_tops[-1]].max()
        if not np.isnan(closest_support):
            plt.plot([double_tops.index[-1], extended_data.index[-1]], [double_tops[-1], closest_support], 'y--', label='Predicted Trend')
    elif not double_bottoms.empty:
        closest_resistance = resistance_levels.index[resistance_levels.index > double_bottoms[-1]].min()
        if not np.isnan(closest_resistance):
            plt.plot([double_bottoms.index[-1], extended_data.index[-1]], [double_bottoms[-1], closest_resistance], 'y--', label='Predicted Trend')

    plt.title(f'Technical Analysis of {symbol}', fontsize=15, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price', fontsize=12, color='white')

    # Adding a grid for better readability
    plt.grid(color='#2C2C2C', linestyle='-', linewidth=0.5)

    # Adjusting legend for better visibility
    legend = plt.legend(loc='upper left', fontsize=10)
    for text in legend.get_texts():
        text.set_color('white')
    img_data = io.BytesIO()
    plt.savefig(img_data,format='png')
    plt.close()
    return base64.b64encode(img_data.getvalue()).decode()

# ---------------------- Lambda Handler Function ----------------------

def lambda_handler(event, context):
    try:
        symbol = event['queryStringParameters']['symbol']
        analysis_type = int(event['queryStringParameters']['analysis_type'])

        data = fetch_historical_data(symbol)
        data['ema_12'] = calculate_ema(data, 12)
        data['ema_26'] = calculate_ema(data, 26)
        data = calculate_bollinger_bands(data, window=20, deviation=2)
        support_levels, resistance_levels = identify_support_resistance(data)
        double_tops, double_bottoms = identify_double_tops_bottoms(data)
        cross_up, cross_down = find_cross_points(data, 'ema_12', 'ema_26')

        if analysis_type == 1:
            image = visualize_basic(data, symbol, double_tops, double_bottoms, support_levels, resistance_levels)
        elif analysis_type == 2:
            image = visualize_adv(data, symbol, double_tops, double_bottoms, support_levels, resistance_levels)
        elif analysis_type == 3:
            image = visualize_pro(data, support_levels, resistance_levels, cross_up, cross_down, double_tops, double_bottoms, symbol)
        else:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid analysis type'})}

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'image/png'},
            'body': image,
            'isBase64Encoded': True
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
