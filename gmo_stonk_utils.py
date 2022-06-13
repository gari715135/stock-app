import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
np.seterr(divide = 'ignore')

class gmostonk:
    def __init__(self, tickers, tickers_df = pd.DataFrame(), saved_locally=True):
        self.tickers = ' '.join(tickers)
        if not saved_locally:
            self.tickers_df = self.get_tickers()
            self.saved_locally = True
        else:
            self.tickers_df = pd.read_csv('stocks.csv')

        dfs = []
        for ticker in tickers:
            dfs.append(self.calculate_returns(ticker))
        dfs = pd.concat(dfs)
        self.tickers_df = dfs.reset_index()

    def get_tickers(self):
        tickers_df = yf.download(self.tickers, group_by='Ticker', start="1998-01-01")
        tickers_df = tickers_df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        self.tickers_df = tickers_df.reset_index()
        tickers_df.to_csv('stocks.csv')
        return tickers_df

    def calculate_returns(self, ticker):
        tickers_df = self.tickers_df.copy().reset_index()
        ticker_df = tickers_df[tickers_df['Ticker'] == ticker]
        ticker_df = ticker_df.set_index(['Date', 'Ticker'])
        ticker_df = ticker_df.select_dtypes(include=["float", 'int'])

        # Calculate period returns and add to df
        log_change = np.log(ticker_df + 10**-5)-np.log(ticker_df.shift(1) + 10**-5)
        returns = log_change
        df_returns = ticker_df.join(returns, rsuffix='_r').dropna()
        ticker_df = df_returns.copy()
        return ticker_df.drop('Volume_r', axis=1).reset_index()

    def OLS_regression(self, ticker1_df, ticker2_df, col1, col2):
        ticker1_df = ticker1_df.set_index('Date')
        ticker2_df = ticker2_df.set_index('Date')

        inter = ticker1_df.index.intersection(ticker2_df.index).values
        ticker1_df = ticker1_df[ticker1_df.index.isin(inter)]
        ticker2_df = ticker2_df[ticker2_df.index.isin(inter)]
        
        # Ordinary Least Squares
        x, y = ticker1_df[col1], ticker2_df[col2]
        col1 = ticker1_df['Ticker'].values[0] + '_' + col1 
        col2 = ticker2_df['Ticker'].values[0] + '_' + col2
        df_final  = pd.DataFrame({col1: x,
                                  col2: y}).dropna()
        reg = 'y~x'
        reg_out = smf.ols(reg, df_final[[col1, col2]])
        reg_fit = reg_out.fit()

        influence = reg_fit.get_influence()
        df_inf = influence.summary_frame()

        df_fits = df_final.join(df_inf)
        #r = df_fits.student_resid
        
        # High leverage rule of thumb. High leverage > 3 * mean(levg)
        leverage = df_fits.hat_diag
        high_leverage_threshold = 3 * np.mean(leverage)

        high_leverage_mask = (abs(leverage) > high_leverage_threshold)
        high_leveraged_data = df_fits[high_leverage_mask]
        normally_leveraged_data = df_fits[~high_leverage_mask]

        x_fit = np.linspace(x.min(), x.max(), len(df_fits))
        y_fit = reg_fit.params[1]*x_fit + reg_fit.params[0]

        return {'Unleveraged Data': normally_leveraged_data,
                'High Leverage Data (HL)': high_leveraged_data,
                'x_fit': x_fit,
                'y_fit': y_fit,
                'reg_fit_summary': reg_fit.summary().as_html()}
