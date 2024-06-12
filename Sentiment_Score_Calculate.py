import pandas as pd
from datetime import timedelta
from datetime import datetime
from enum import Enum
# 將同一天的新聞情緒量化取平均

# 獲取當前日期和時間
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

# 定義一個枚舉類
class Stock(Enum):
    Tesla = 1
    Microsoft = 2
    Amazon = 3
    Test = 4

# [自行更改] 設定讀取股票名稱
TARGET = Stock.Amazon

# [自行更改] 設定讀取股票期間
Stock_Period = '2015-2020'
# Stock_Period = '2023.07-2023.12'

# 依不同股票名稱，讀取不同檔案
if (TARGET == Stock.Tesla):
    # 原始股價路徑
    csv_path = f'./Stock_Datas/{Stock_Period}/TSLA.csv' 
    # 情緒分析輸入檔案路徑
    sentiment_csv_path = f'./Output_News/Tesla/{Stock_Period}_Tesla_News_with_ChatGPT4.csv'  # 請替換為您的檔案路徑
    # 情緒分析量化後輸出檔案路徑
    output_csv_path = f'./Stock_Datas/{Stock_Period}/{formatted_now}_TSLA_with_Sentiment.csv'  # 請替換為您想要儲存的檔案路徑
elif (TARGET == Stock.Microsoft):
    # 原始股價路徑
    csv_path = f'./Stock_Datas/{Stock_Period}/MSFT.csv' 
    # 情緒分析輸入檔案路徑
    sentiment_csv_path = f'./Output_News/Microsoft/{Stock_Period}_Microsoft_News_with_ChatGPT4.csv'   # 請替換為您的檔案路徑
    # 情緒分析量化後輸出檔案路徑
    output_csv_path = f'./Stock_Datas/{Stock_Period}/{formatted_now}_MSFT_with_Sentiment.csv'  # 請替換為您想要儲存的檔案路徑
elif (TARGET == Stock.Amazon):
    # 原始股價路徑
    csv_path = f'./Stock_Datas/{Stock_Period}/AMZN.csv' 
    # 情緒分析輸入檔案路徑
    sentiment_csv_path = f'./Output_News/2024_05_06-23_08_28_ChatGPT_Results_ChatGPT4.csv'  # 請替換為您的檔案路徑
    # sentiment_csv_path = f'./Output_News/Amazon/{Stock_Period}_Amazon_News_with_ChatGPT4.csv'  # 請替換為您的檔案路徑
    # 情緒分析量化後輸出檔案路徑
    output_csv_path = f'./Stock_Datas/{Stock_Period}/{formatted_now}_AMZN_with_Sentiment.csv'  # 請替換為您想要儲存的檔案路徑
else:
    print ('讀取檔案，路徑異常!')

def merge_previous_day_sentiment_with_tsla(csv_path, sentiment_csv_path, output_csv_path):
    # 讀取csv和情緒數據
    data = pd.read_csv(csv_path)
    try:
        sentiment_data = pd.read_csv(sentiment_csv_path)
    except UnicodeDecodeError:
        sentiment_data = pd.read_csv(sentiment_csv_path, encoding='ISO-8859-1')

    # 將情緒映射到分數
    sentiment_scores = {'BULLISH': 100, 'NEUTRAL': 0, 'BEARISH': -100}
    sentiment_data['Score'] = sentiment_data['Sentiment'].map(sentiment_scores)

    # 處理日期
    date_column = 'Published Date' if 'Published Date' in sentiment_data.columns else 'Date'
    sentiment_data[date_column] = pd.to_datetime(sentiment_data[date_column]).dt.date

    # 計算每日情緒分數
    daily_sentiment = sentiment_data.groupby(date_column)['Score'].mean().reset_index()
    daily_sentiment.rename(columns={date_column: 'Date', 'Score': 'Daily Sentiment'}, inplace=True)

    # 確保數據中的日期格式一致
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    # 合併情緒分數到數據中，並計算前一天與當天的平均情緒分數
    merged_data = pd.merge(data, daily_sentiment, on='Date', how='left')
    merged_data['Previous Day'] = merged_data['Date'] - timedelta(days=1)
    merged_data = pd.merge(merged_data, daily_sentiment[['Date', 'Daily Sentiment']], left_on='Previous Day', right_on='Date', how='left', suffixes=('', '_Prev'))
    # 前一天與當天的平均情緒分數
    # merged_data['Average Sentiment'] = merged_data[['Daily Sentiment', 'Daily Sentiment_Prev']].mean(axis=1)
    # 當天的平均情緒分數
    merged_data['Average Sentiment'] = merged_data['Daily Sentiment']

    # 使用fillna方法将'Average Sentiment'中的NaN值替换为0
    merged_data['Average Sentiment'] = merged_data['Average Sentiment'].fillna(0)

    # 選取需要的列
    merged_data = merged_data[['Date', 'Average Sentiment'] + [col for col in data.columns if col != 'Date']]

    # 儲存合併後的數據到新的CSV檔案
    merged_data.to_csv(output_csv_path, index=False)
    print(f"已將合併後的數據儲存至 {output_csv_path}")


# 執行函數
merge_previous_day_sentiment_with_tsla(csv_path, sentiment_csv_path, output_csv_path)
