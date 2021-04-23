# Imports
import json

import numpy as np
import pandas as pd
import pandas_ta as ta  # Pandas Technical Analysis for easily adding technical indicators
import plotly.graph_objects as go  # Plotly - Add various types of graph objects
import yfinance as yf  # Yahoo! Finance for equity information and OHLCV data
from plotly.subplots import make_subplots  # Plotly - Add multiple plots to one figure
# import bt   # Backtesting to evaluate returns using various strategies


# Dump dictionary to JSON file
def dict_to_json(py_dict, filename):
    with open(filename, 'w') as fp:
        json.dump(py_dict, fp, indent=4)


# Merge a list of data frames into a single data frame with a multi index for the columns
def merge_dfs(df_dict):
    df_list = []  # Empty list to hold modified data frames
    for eq in df_dict.keys():
        # Change columns to multi index
        df_dict[eq].columns = pd.MultiIndex.from_product([[eq], df_dict[eq].columns])
        # Store to be merged later
        df_list.append(df_dict[eq])
    # Re-combine into one df with MultiIndex as columns
    return pd.concat(df_list, axis=1)


# Find points of intersection between two series
def find_intersections(a, b, up_name, down_name):
    diff = a - b
    return np.select([((diff > 0) & (diff.shift() < 0)),
                      ((diff < 0) & (diff.shift() > 0))],
                     [up_name, down_name], None)


# Create relative performance chart of all stocks
def relative_performance_chart(df, col_name):
    # Create graph object
    fig = go.Figure()
    fig.update_layout(title={'text': 'Relative Percent Return'})
    fig.update_yaxes(title={'text': 'Percent Return'})

    # Loop through each equity in the data frame
    for eqty, eqty_df in df.groupby(level=0, axis=1):
        # Add line to the graph
        fig.add_trace(go.Scatter(x=eqty_df.index, y=eqty_df[eqty][col_name], name=eqty))

    # Save and show figure
    fig.write_html('HTMLs\\' + col_name + '.html', auto_open=True)


# Plot each stock's candlestick chart individually
def technical_analysis_graph(name, graph_df):
    # Create graph object
    fig = make_subplots(rows=3, cols=1, row_heights=[0.6, 0.2, 0.2], shared_xaxes=True, vertical_spacing=0.02)
    fig.update_layout(title={'text': name + ' Technical Analysis'},
                      xaxis1_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=list(graph_df.index) + list(graph_df.index[::-1]),
                             y=list(graph_df['BBU']) + list(graph_df['BBL'][::-1]),
                             fill='toself', fillcolor='rgba(0, 176, 246, 0.2)',
                             name='Bollinger Bands'),
                  row=1, col=1)
    
    # Add candlestick plot of price data
    fig.add_trace(go.Candlestick(x=graph_df.index, open=graph_df['open'], high=graph_df['high'],
                                 low=graph_df['low'], close=graph_df['close'], name=name))

    # Add SMAs
    fig.add_trace(go.Scatter(x=graph_df.index, y=graph_df['SMA50'], name='SMA50', line=dict(color='yellow')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=graph_df.index, y=graph_df['SMA200'], name='SMA200', line=dict(color='purple')),
                  row=1, col=1)

    # Add golden crosses & death crosses
    fig.add_trace(go.Scatter(x=graph_df.index[graph_df[sma_cross_col_name] == sma_cross_buy],
                             y=graph_df['close'][graph_df[sma_cross_col_name] == sma_cross_buy],
                             name=sma_cross_buy, mode='markers',
                             marker=dict(symbol='arrow-bar-up', size=10, color='green')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=graph_df.index[graph_df[sma_cross_col_name] == sma_cross_sell],
                             y=graph_df['close'][graph_df[sma_cross_col_name] == sma_cross_sell],
                             name=sma_cross_sell, mode='markers',
                             marker=dict(symbol='arrow-bar-down', size=10, color='red')),
                  row=1, col=1)

    # Add MACD sub-plot
    fig.add_trace(go.Scatter(x=graph_df.index, y=graph_df['MACD'], line=dict(color='blue', width=1),
                             name='MACD', showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=graph_df.index, y=graph_df['MACD_S'], line=dict(color='orange', width=1),
                             name='MACD Signal', showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=graph_df.index[graph_df[macd_cross_col_name] == macd_cross_buy],
                             y=graph_df['MACD'][graph_df[macd_cross_col_name] == macd_cross_buy],
                             name=macd_cross_buy, mode='markers',
                             marker=dict(symbol='arrow-up', size=6, color='green')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=graph_df.index[graph_df[macd_cross_col_name] == macd_cross_sell],
                             y=graph_df['MACD'][graph_df[macd_cross_col_name] == macd_cross_sell],
                             name=macd_cross_sell, mode='markers',
                             marker=dict(symbol='arrow-down', size=6, color='red')),
                  row=2, col=1)

    # Add RSI sub-plot
    fig.add_trace(go.Scatter(x=graph_df.index, y=graph_df['RSI'], line=dict(color='blue', width=1),
                             name='RSI', showlegend=False),
                  row=3, col=1)
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor='green', opacity=0.2, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor='red', opacity=0.2, row=3, col=1)

    # Save and show figure
    fig.write_html('HTMLs\\' + name + '.html', auto_open=True)


# Let's make some money
if __name__ == '__main__':
    equity_list = ['IGV', 'XLK']
    # equity_list = ['SPY', 'XLI', 'XLV', 'XLK', 'XLF', 'XLB', 'XLE', 'XLU', 'XLY', 'XLP', 'XLC', 'XLRE']  # Sectors
    # equity_list = ['BTC-USD', 'ETH-USD', 'DOGE-USD']    # Crypto
    (sma_cross_col_name, sma_cross_buy, sma_cross_sell) = ('SMA50-SMA200', 'Golden Cross', 'Death Cross')
    (macd_cross_col_name, macd_cross_buy, macd_cross_sell) = ('MACD Crossover', 'MACD Buy', 'MACD Sell')

    # Define Technical Analysis Strategy
    SMA_BB_MACD = ta.Strategy(
        name='SMAs, BBs, and MACD',
        description='Short & Long Term SMA. BBs. MACD. RSI.',
        ta=[
            {'kind': 'sma', 'length': 50, 'col_names': ('SMA50',)},
            {'kind': 'sma', 'length': 200, 'col_names': ('SMA200',)},
            {'kind': 'bbands', 'length': 20, 'col_names': ('BBL', 'BBM', 'BBU', 'BBB')},
            {'kind': 'macd', 'fast': 8, 'slow': 21, 'col_names': ('MACD', 'MACD_H', 'MACD_S')},
            {'kind': 'rsi', 'col_names': ('RSI',)},
        ]
    )

    equity_info_dict = {}  # Empty dict to hold information about equities
    equity_data_dict = {}  # Empty dict to hold data frames of each equity
    for e, equity in enumerate(equity_list):  # Loop through each equity
        # Print logging
        print('=' * 75)
        print_prefix = 'Equity ' + str(e + 1) + ' of ' + str(len(equity_list)) + ' - ' + equity + ' - '

        # Pull basic equity information from Yahoo! Finance
        print(print_prefix + 'Retrieving information from Yahoo! Finance')
        equity_info_dict[equity] = yf.Ticker(equity).info

        # Pull data frame of open, high, low, close, adj_close, and volume from Yahoo! Finance
        print(print_prefix + 'Downloading OHLCV data from Yahoo! Finance')
        equity_df = yf.download(tickers=equity, group_by='ticker', progress=False,
                                period='3mo',       # Options = 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                                interval='1h')     # Options = 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        equity_df.dropna(how='all', inplace=True)

        # Add technical indicators
        print(print_prefix + 'Adding technical indicators')
        equity_df['Percent Return'] = ((equity_df['Close'] / equity_df['Close'][0]) - 1) * 100
        equity_df.ta.strategy(SMA_BB_MACD, verbose=False)

        # Identify trading signals
        print(print_prefix + 'Identifying trading signals')
        equity_df[sma_cross_col_name] = find_intersections(equity_df['SMA50'], equity_df['SMA200'],
                                                           sma_cross_buy, sma_cross_sell)
        equity_df[macd_cross_col_name] = find_intersections(equity_df['MACD'], equity_df['MACD_S'],
                                                            macd_cross_buy, macd_cross_sell)

        # Create technical analysis graph
        print(print_prefix + 'Graphing equity data')
        technical_analysis_graph(equity, equity_df)

        # Store data frame in list to be consolidated later
        equity_data_dict[equity] = equity_df
        print('=' * 75)

    all_equities_df = merge_dfs(equity_data_dict)  # Combine individual dfs into a single df w/ multi index columns
    all_equities_df.to_csv('TechnicalIndicators.csv')   # Store technical analysis data in a CSV
    dict_to_json(equity_info_dict, 'EquityInfo.json')   # Store equity information in a JSON file
    relative_performance_chart(all_equities_df, 'Percent Return')
