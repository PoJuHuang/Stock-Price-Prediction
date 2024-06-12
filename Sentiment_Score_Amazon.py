import pandas as pd

# 載入數據集
news_df = pd.read_csv('./Output_News/Amazon/2020.05.08-2020.09.11_Amazon_News_with_ChatGPT4.csv', encoding='ISO-8859-1')
amzn_df = pd.read_csv('./Stock_Datas/2020_AMZN_5min/2020_AMZN_5min.csv', encoding='ISO-8859-1')

# 轉換 news_df 中的 'Published Date' 為時區不敏感的 datetime 對象
# 這是為了解決在比較時區敏感(datetime with timezone)和時區不敏感(datetime without timezone)對象時發生的錯誤
news_df['Published Date'] = pd.to_datetime(news_df['Published Date'], errors='coerce').dt.tz_localize(None)

# 確保 amzn_df 中的 'Date' 也是時區不敏感的
# 這步操作確保兩個數據集中用於比較的 datetime 對象都是時區不敏感的，從而避免因時區差異導致的錯誤
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'], errors='coerce').dt.tz_localize(None)

# 準備數據
news_df['Published Date'] = pd.to_datetime(news_df['Published Date'])
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'])

# 對數據進行排序
news_df.sort_values('Published Date', inplace=True)
amzn_df.sort_values('Date', inplace=True)

# 定義將情緒文字轉換為數值的函數
def sentiment_to_numeric(sentiment):
    return {'BULLISH': 100, 'NEUTRAL': 0, 'BEARISH': -100}.get(sentiment, 0)

# 應用情緒分數轉換
news_df['Sentiment_Score'] = news_df['Sentiment'].apply(sentiment_to_numeric)

# 初始化一個空的DataFrame用於存儲結果
merged_df = amzn_df.copy()

# 對於每條新聞發布日期，找到最接近的股票時間點並合併情緒分數
for index, row in news_df.iterrows():
    nearest_date = amzn_df[amzn_df['Date'] >= row['Published Date']].head(1)['Date']
    if not nearest_date.empty:
        merged_df.loc[merged_df['Date'] == nearest_date.values[0], 'Sentiment_Score'] = row['Sentiment_Score']

# 填充缺失的情緒分數為0
merged_df['Sentiment_Score'].fillna(0, inplace=True)

# 輸出文件路徑
output_path = './Stock_Datas/2020_AMZN_5min/Merged_AMZN_with_Sentiment.csv'
# 將結果保存到CSV文件
merged_df.to_csv(output_path, index=False)
print(f'文件已保存至 {output_path}')
