#MARKET SCREENER SINGLE

import numpy as np
import pandas as pd
import datetime 
import ta
import os
from dotenv import load_dotenv
load_dotenv()

from def_symbols_tv import get_symbol_names

import requests
import json

#ff_fundamenal ---------------------------
from bs4 import BeautifulSoup
from selenium import webdriver

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
#ff_fundamenal ---------------------------


# import technical libraries
import ta
from ta.trend import EMAIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.trend import CCIIndicator
from ta.trend import MACD
from ta.momentum import StochRSIIndicator
from ta.momentum import AwesomeOscillatorIndicator
from ta.momentum import williams_r
from ta.momentum import UltimateOscillator
from ta.momentum import ROCIndicator

from tvDatafeed import TvDatafeed, Interval

#username = 'contactus@xaviermcallister.com'
#password = 'xaviermcallister2019!!'


class MarketWatch():
    def __init__(self, exchange='OANDA'):

        #datafeed log in 
        USERNAME = os.getenv('USERNAME')
        PASSWORD = os.getenv('PASSWORD')
        
        self.tv = TvDatafeed(username=USERNAME, password=PASSWORD)
        self.exchange = exchange
        self.symbols = get_symbol_names()
        self.token = os.getenv('FOREXAPINEWS_TOKEN')
        
    def pull_data(self, symbol, timeframe):
        
        df = self.tv.get_hist(symbol=symbol,exchange=self.exchange ,interval=timeframe, n_bars=500) #Interval.in_1_hour
        # create DataFrame out of the obtained data
        df = pd.DataFrame(df)
        # convert time in seconds into the datetime format
        df['time']=pd.to_datetime(df.index, unit='s')
        df.index = df.time.values
        df = df.drop(["time", "symbol", "volume"], axis = 1) #"open", "high", "low"
        df = df.rename(columns = {"open": "Open", 
                         "close": "Close",
                         "high": "High",
                         "low": "Low"})
        df = df.dropna()
        self.df = df
        
        return df
    
    
    def fib_levels(self, df):
        
        df = df[-200:]
        highest_swing = -1
        lowest_swing = -1
        for i in range(1,df.shape[0]-1):
            if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and (highest_swing == -1 or df['High'][i] > df['High'][highest_swing]):
                highest_swing = i

            elif df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and (lowest_swing == -1 or df['Low'][i] < df['Low'][lowest_swing]):
                lowest_swing = i

        ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
        colors = ["black","r","g","b","cyan","magenta","yellow"]
        levels = []
        max_level = df['High'][highest_swing]
        min_level = df['Low'][lowest_swing]
        for ratio in ratios:
            if highest_swing > lowest_swing: # Uptrend
                levels.append(max_level - (max_level-min_level)*ratio)
            else: # Downtrend
                levels.append(min_level + (max_level-min_level)*ratio)
        levels = np.around(levels, 5)
        
        return levels
    
    def __indicators(self, df):
        # roc
        roc = ROCIndicator(close=df.Close, window=5)
        df["roc"] = roc.roc()

        return df

    def strength_meter(self):
        symbols_ = ["AUDUSD", "NZDUSD", "EURUSD", "GBPUSD"]
        symbols_2 = ["USDCAD", "USDCHF", "USDJPY"]
        symbols = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
        data__ = []

        for symb in symbols_:
            data = self.pull_data(symb, Interval.in_4_hour)
            data = self.__indicators(data)
            data__.append(round(data.roc[-1], 2))

        for symb in symbols_2:
            data = self.pull_data(symb, Interval.in_4_hour)
            data = self.__indicators(data)
            data__.append(round(-data.roc[-1], 2))

        data__.append(0)

        roc_data = pd.DataFrame()
        roc_data["symbol"] = symbols
        roc_data["strength"] = data__
        roc_data.sort_values("strength", ascending=False, inplace=True)
        roc_data.reset_index(drop=True, inplace=True)

        return roc_data
    
    def tops(self):

        time_frame = Interval.in_daily
        signals = []
        dataset = []
        final_data = []

        for symbol in self.symbols:
            df = self.pull_data(symbol, time_frame)
            #print(df)
            df["%change"] = round(df.Close.pct_change(), 4)*100
            dataset.append(df)

        for symbol, df in zip(self.symbols, dataset):
            dff = pd.DataFrame(df.iloc[-1]).T
            dff.index = [symbol]
            final_data.append(dff)

        table = pd.concat(final_data)
        table.reset_index(inplace = True)
        table = table.rename(columns = {"index":"Currency Pairs", "Close" : "Price"})
        table = table[['Currency Pairs', 'Price', '%change']]

        m_df = table[table['%change'] > 0]
        m_df = m_df.sort_values(by='%change', ascending=False)[:5] #top 5 buyers
        m_df["%change"] = [round(i, 2) for i in m_df["%change"]]
        m_df["%change"] = [f"{i}%" for i in m_df["%change"]]
        #m_df = m_df.drop(["Price"], axis = 1)

        m_dfs = table[table['%change'] < 0]
        m_dfs = m_dfs.sort_values(by='%change', ascending=True)[:5] #top5 sellers
        m_dfs["%change"] = [round(i, 2) for i in m_dfs["%change"]]
        m_dfs["%change"] = [f"{i}%" for i in m_dfs["%change"]]
        #m_dfs = m_dfs.drop(["Price"], axis = 1)

        return m_df, m_dfs
    
    def __add_indicators(self, df):

        #----------------------------------------------------------------
        #Exponential Moving Averages
        #----------------------------------------------------------------

        #ema5
        ema5 = EMAIndicator(close = df.Close, window = 5)
        df["ema5"] = round(ema5.ema_indicator(), 5)

        #ema10
        ema10 = EMAIndicator(close = df.Close, window = 10)
        df["ema10"] = round(ema10.ema_indicator(), 5)

        #ema20
        ema20 = EMAIndicator(close = df.Close, window = 20)
        df["ema20"] = round(ema20.ema_indicator(), 5)

        #ema30
        ema30 = EMAIndicator(close = df.Close, window = 30)
        df["ema30"] = round(ema30.ema_indicator(), 5)

        #ema50
        ema50 = EMAIndicator(close = df.Close, window = 50)
        df["ema50"] = round(ema50.ema_indicator(), 5)

        #ema100
        ema100 = EMAIndicator(close = df.Close, window = 100)
        df["ema100"] = round(ema100.ema_indicator(), 5)

        #ema200
        ema200 = EMAIndicator(close = df.Close, window = 200)
        df["ema200"] = round(ema200.ema_indicator(), 5)


        #----------------------------------------------------------------
        # Simple Moving Averages
        #----------------------------------------------------------------

        #sma5
        sma5 = SMAIndicator(close = df.Close, window = 5)
        df["sma5"] = round(sma5.sma_indicator(), 5)

        #sma10
        sma10 = SMAIndicator(close = df.Close, window = 10)
        df["sma10"] = round(sma10.sma_indicator(), 5)

        #sma20
        sma20 = SMAIndicator(close = df.Close, window = 20)
        df["sma20"] = round(sma20.sma_indicator(), 5)

        #sma30
        sma30 = SMAIndicator(close = df.Close, window = 30)
        df["sma30"] = round(sma30.sma_indicator(), 5)

        #sma50
        sma50 = SMAIndicator(close = df.Close, window = 50)
        df["sma50"] = round(sma50.sma_indicator(), 5)

        #sma100
        sma100 = SMAIndicator(close = df.Close, window = 100)
        df["sma100"] = round(sma100.sma_indicator(), 5)

        #sma200
        sma200 = SMAIndicator(close = df.Close, window = 200)
        df["sma200"] = round(sma200.sma_indicator(), 5)

        #----------------------------------------------------------------
        # Oscilators
        #----------------------------------------------------------------

        #rsi(14)
        rsi = RSIIndicator(close = df.Close, window = 14)
        df["rsi"] = round(rsi.rsi(), 5)

        #stoch(14,3)
        stoch = StochasticOscillator(df.High, df.Low, df.Close, window = 14, smooth_window = 3)
        df["stoch"] = round(stoch.stoch_signal(), 5)

        #adx(14)
        adx = ADXIndicator(df.High, df.Low, df.Close, window = 14)
        df["adx"] = round(adx.adx(), 5)

        #cci
        cci = CCIIndicator(df.High, df.Low, df.Close, window = 20)
        df["cci"] = round(cci.cci(), 5)

        """#momentum
        df["mom"] = talib.MOM(df.Close, timeperiod=10)"""

        #AO
        ao = AwesomeOscillatorIndicator(df.High, df.Low, window1 = 5, window2 = 34)
        df["ao"] = round(ao.awesome_oscillator(), 5)

        #stochrsi(14,3,3)
        stochrsi = StochRSIIndicator(df.Close, window = 14, smooth1 = 3, smooth2 = 3)
        df["stochrsi"] = round(stochrsi.stochrsi_k(), 5)

        #MACD(12,26)
        macd = MACD(df.Close)
        df["macd"] = round(macd.macd_diff(), 5)

        #WPR(14)
        wpr = williams_r(df.High, df.Low, df.Close)
        df["wpr"] = round(wpr, 5)

        #UO(14)
        uo = UltimateOscillator(df.High, df.Low, df.Close)
        df["uo"] = round(uo.ultimate_oscillator(), 5)

        df.dropna(inplace = True)
        return df
    
    def __confluence(self, df):
        signal_buy = 0
        signal_sell = 0
        signal_neutral = 0

        indicators_list = ["EMA5", "EMA10", "EMA20", "EMA30", "EMA50", "EMA100", 
                          "SMA5", "SMA10", "SMA20", "SMA30", "SMA50", "SMA100",
                          "RSI(14)", "STOCH(14, 3, 3)", "CCI(20)", "ADX(14)", "MOM(10)", "AO",
                           "STOCHRSI(3, 3, 14)", "MACD(12,26)","WPR(14)", "ULTIMATE OSCILLATOR(7,14,28)"]

        indicator_signal = []

        #----------------------------------------------------------------
        #Exponential Moving Averages
        #----------------------------------------------------------------

        if df["ema5"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema5"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["ema10"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema10"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["ema20"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema20"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["ema30"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema30"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["ema50"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema50"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["ema100"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["ema100"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        #----------------------------------------------------------------
        #Simple Moving Averages
        #----------------------------------------------------------------

        if df["sma5"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma5"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["sma10"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma10"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["sma20"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma20"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["sma30"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma30"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal == "Neutral"
            signal_neutral +=1 

        indicator_signal.append(signal)

        if df["sma50"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma50"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["sma100"][-1] < df["Close"][-1]:
            signal = "buy"
            signal_buy += 1

        elif df["sma100"][-1] > df["Close"][-1]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        #----------------------------------------------------------------
        #Oscilators
        #----------------------------------------------------------------

        if df["rsi"][-1] > 70.0:
            signal = "buy"
            signal_buy += 1

        elif df["rsi"][-1] < 30.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["stoch"][-1] > 80.0:
            signal = "buy"
            signal_buy += 1

        elif df["stoch"][-1] < 20.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["cci"][-1] > 100.0:
            signal = "buy"
            signal_buy += 1

        elif df["cci"][-1] < -100.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["adx"][-1] > 25.0:
            signal = "buy"
            signal_buy += 1

        elif df["adx"][-1] > 25.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        """if df["mom"][-1] > 0.0:
            signal = "buy"
            signal_buy += 1

        elif df["mom"][-1] < 0.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)"""

        if df["ao"][-1] > 0.0:
            signal = "buy"
            signal_buy += 1

        elif df["ao"][-1] < 0.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["stochrsi"][-1] > 80.0:
            signal = "buy"
            signal_buy += 1

        elif df["stochrsi"][-1] < 20.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["macd"][-1] > df["macd"][-2]:
            signal = "buy"
            signal_buy += 1

        elif df["macd"][-1] < df["macd"][-2]:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["wpr"][-1] > -20.0:
            signal = "buy"
            signal_buy += 1

        elif df["wpr"][-1] < -80.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        if df["uo"][-1] > 70.0:
            signal = "buy"
            signal_buy += 1

        elif df["uo"][-1] < 30.0:
            signal = "sell"
            signal_sell += 1

        else:
            signal = "neutral"
            signal_neutral += 1

        indicator_signal.append(signal)

        #----------------------------------------------------------------
        #Confluence
        #----------------------------------------------------------------

        total_signals = sum([signal_buy, signal_sell, signal_neutral])

        perc_buy = round((signal_buy/total_signals) * 100, 1)
        perc_sell = round((signal_sell/total_signals) * 100, 1)
        perc_neutral = round((signal_neutral/total_signals) * 100, 1)

        #print(f"{perc_buy}% Buy, {perc_sell}% Sell, {perc_neutral}% Neutral")

        #----------------------------------------------------------------
        #Table
        #----------------------------------------------------------------

        table = pd.DataFrame([f"{perc_buy}%", f"{perc_sell}%", f"{perc_neutral}%"])
        table = table.T
        table.columns = ["Buy", "Sell", "Neutral"]
        #table

        if perc_buy > 75:
            con_signal = "Strong Buy"

        elif perc_buy > 49 < 75:
            con_signal = "Buy"

        elif perc_sell > 75:
            con_signal = "Strong Sell"

        elif perc_sell > 49 < 75:
            con_signal = "Sell"

        elif perc_neutral > 49 or (perc_buy < 50 and perc_sell < 50):
            con_signal = "Neutral"

        sig_data = [con_signal, (f"{perc_buy}% Buy") , (f"{perc_sell}% Sell"), f"{perc_neutral}% Neutral"]
        sig_df = pd.DataFrame(sig_data[1:])
        sig_df.columns = [sig_data[0]]

        return sig_df


    def screener(self, symbol, time_frame):

        time_frame = time_frame
        df = self.pull_data(symbol, time_frame)
        data = self.__add_indicators(df)
        conf = self.__confluence(data)

        return [conf, df]
    
    def forexnews_pair(self, pair, item = 5):
        token=self.token
        no_item = item
        currencypair = f'{pair[0:3]}-{pair[3:]}'
        try:
            
            url = f"https://forexnewsapi.com/api/v1?currencypair={currencypair}&items={no_item}&page=1&token={token}"
            r = requests.get(url) #, headers=headers)
            data = BeautifulSoup(r.text, 'html.parser')
            data = json.loads(str(data))
            data = pd.DataFrame.from_dict(data['data'])
            data = data.drop(['news_url', 'image_url', 'topics', 'source_name'], axis = 1)
            
        except:
            data = None
            
        return data
    
    def general_forexnews(self, items = 5):

        try:
            token=self.token
            no_item = items

            url = f"https://forexnewsapi.com/api/v1/category?section=general&items={no_item}&page=1&token={token}"
            #url = f"https://forexnewsapi.com/api/v1?currencypair={currencypair}&items={no_item}&page=1&token={token}"
            r = requests.get(url) #, headers=headers)
            data = BeautifulSoup(r.text, 'html.parser')
            data = json.loads(str(data))
            data = pd.DataFrame.from_dict(data['data'])
            data = data.drop(['news_url', 'image_url', 'topics', 'source_name'], axis = 1)
        except:
            data = None
        return data
    
    
    def ff_fundamentals(self, deploy = True):
        #working data with effect - daily
        impact = []

        url = 'https://www.forexfactory.com/calendar?day=today'
        
        if deploy: 
            driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), 
                              desired_capabilities=DesiredCapabilities.CHROME, chrome_options=options)
        else: 
        
            driver = webdriver.Chrome(service = Service('chromedriver.exe'))
        
        driver.get(url)
        bs = BeautifulSoup(driver.page_source,"lxml")
        table = bs.find("table", class_ = 'calendar__table')

        list_of_rows = []
        links = []
        #Filtering events that have a href link
        for row in table.find_all('tr', {'data-eventid':True}):
            list_of_cells = []
            #Filtering high-impact events
            for span in row.find_all('span', class_=['icon icon--ff-impact-yel', 'icon icon--ff-impact-gra',\
                                                     'icon icon--ff-impact-red', 'icon icon--ff-impact-ora']):
                #links.append(url + "#detail=" + row['data-eventid'])

                impact.append(span['title'])
                #Extracting the values from the table data in each table row
                for cell in row.find_all('td', class_=[
                  'calendar__cell calendar__date date', 
                  'calendar__cell calendar__time time',
                  'calendar__cell calendar__currency currency',
                  "calendar__impact-icon calendar__impact-icon--screen",
                  #'calendar__cell calendar__impact impact', 
                  'calendar__cell calendar__event event', 
                  'calendar__cell calendar__detail detail',
                  'calendar__cell calendar__actual actual', 
                  'calendar__cell calendar__forecast forecast', 
                  'calendar__cell calendar__previous previous',
                  'full calendarspecs__specdescription']):


                    list_of_cells.append(cell.text)
                list_of_rows.append(list_of_cells)

        df = pd.DataFrame(list_of_rows, columns =
        ['Date', 'Time', 'Country','Event Title','None', 'Actual', 'Forecast', 'Previous'])

        df.drop("None", axis = 1, inplace = True)
        country = []
        for i in df.Country.str.split('\n'):
            count = i[1]
            country.append(count)

        #process time column
        t_time = []
        for i in df.Time: #.str.split('\n'):
            if i == '   ':
                tt = np.NaN

            else:
                tt = i.split('\n')
                if len(tt) > 1:
                    tt = tt[1]

                else: 
                    tt = i

            t_time.append(tt) 

        #process date column
        d_date = []
        for i in df.Date: #.str.split('\n'):
            if i == ' ':
                tt = np.NaN

            else:
                tt = i
                tt = tt[0:4] + ' ' + tt[4:10]

            d_date.append(tt) 

        #process previous column 
        prev = []
        for i in df.Previous: #.str.split('\n'):
            if i == '\n':
                pv = np.NaN

            else:
                try:
                    pv = i.split('\n')
                    if len(pv) > 1:
                        pv = pv[1]

                    else: 
                        pv = i

                except: 
                    pv = np.NaN

            prev.append(pv)

        df["Country"] = country
        df["Time"] = t_time
        df["Time"] = df.Time.ffill(axis = 0)
        df["Date"] = d_date
        df["Date"] = df.Date.ffill(axis = 0)
        df['Previous'] = prev
        df["Impact"] = impact

        return df