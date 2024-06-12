import json
import requests
import pandas as pd
import numpy as np
import datetime
import os
from enum import Enum

# 輸出檔案路徑
savePath = "Output_ChatGPT"
# 設置你的OpenAI ChatGPT API密鑰Key
API_KEY = ''
# API網址
ENDPOINT = "https://api.openai.com/v1/chat/completions"
# 設定GPT模型
# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4-0125-preview"

# 定義一個枚舉類
class Stock(Enum):
    Tesla = 1
    Microsoft = 2
    Amazon = 3
    Test = 4

# 定義目標商品，自行更改Tesla，Microsoft，Amazon or Test
Target = Stock.Test

# 決定讀取檔案位置
file_path = ''
if (Target == Stock.Tesla):
    # 定義讀取CSV檔案的路徑，請替換成你的檔案路徑
    # file_path = './Output_News/Tesla/2015-2020_Tesla_News.csv' 
    file_path = './Output_News/Tesla/2023.07-2023.12_Tesla_News.csv'  
elif (Target == Stock.Microsoft):
    # file_path = './Output_News/Microsoft/2015-2020_Microsoft_News.csv'
    file_path = './Output_News/Microsoft/2023.07-2023.12_Microsoft_News.csv'
elif (Target == Stock.Amazon):
    # file_path = './Output_News/Amazon/2015-2020_Amazon_News.csv'
    # file_path = './Output_News/Amazon/2023.07-2023.12_Amazon_News.csv'
    file_path = './Output_News/Amazon/2020_Amazon_Google_News.csv'
    # file_path = './Output_News/2024_03_28_09_12_31_Amazon_News_Results.csv'
elif (Target == Stock.Test):
    file_path = './Output_News/2024_05_06_23_05_46_Amazon_News_Results.csv'
else:
    print (f'讀取路徑{file_path}異常!')

# 初始化對話歷史列表，第一次請求包含原始prompt
# chat_history = [
#     {"role": "system", "content": "You will work as a Sentiment Analysis for Financial news. I will share news Title and Snippet. You will only answer as:\n\n BEARISH,BULLISH,NEUTRAL. No further explanation. \n Got it?"}
# ]d
chat_system_prompt = {"role": "system", "content": "You will work as a Sentiment Analysis for Financial news. I will share news Title and Snippet. You will only answer as:\n\n BEARISH,BULLISH,NEUTRAL. No further explanation. \n Got it?"}

def chat_with_openai(title,snippet):
    global chat_history
  
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    content = "Title: " + str(title) + "." + "Snippet: " + str(snippet)
    # chat_history.append({"role": "user", "content": content})
    chat_new = []
    chat_new.append(chat_system_prompt)
    chat_new.append({"role": "user", "content": content})
    data = {
        "model": MODEL,
        "messages": chat_new
    }
    
    response = requests.post(ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        # 從回應中提取助理的回答並新增到對話歷史
        answer = response.json()['choices'][0]['message']['content']
        # chat_history.append({"role": "assistant", "content": answer})
        return answer
    else:
        return "Error: " + response.text

def process_excel_file(file_path):
    '''從Excel讀取檔案'''
    try:
        df = pd.read_excel(file_path)
        df['Sentiment'] = ''  # Initialize Sentiment column
        
        for index, row in df.iterrows():
            sentiment = chat_with_openai(row['Title'], row['Snippet'])
            df.at[index, 'Sentiment'] = sentiment
            save_partial_results(df, file_path)
        
        save_partial_results(df, file_path)
        
    except Exception as e:
        print(f'Error processing the file: {e}')

def process_csv_file(file_path):
    '''從CSV讀取檔案'''
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df['Sentiment'] = ''  # Initialize Sentiment column
        
        for index, row in df.iterrows():
            # 檢查Title是否存在並設定適當值
            title = row.get('Title', '')  # 如果Title不存在，則設為空字符串
            sentiment = chat_with_openai(title,row['Snippet'])
            df.at[index, 'Sentiment'] = sentiment
            
            save_partial_results(df, file_path)
            # # Save every 10 records
            # if index % 10 == 9:
            #     save_partial_results(df, file_path)
                
        # Save final results after loop
        save_partial_results(df, file_path)
        
    except Exception as e:
        print(f'Error processing the file: {e}')

count = 0
def save_partial_results(df, original_file_path):
    """
    Save the DataFrame to a new CSV file, appending a timestamp to the filename to avoid overwriting.
    """
    # 例如：'2024_01_30_15_30_00' 格式
    current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    output_dir = savePath
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_file_path = os.path.join(output_dir, f"{current_time}_ChatGPT_Results.csv")
    
    df.to_csv(new_file_path, index=False, encoding='ISO-8859-1')
    global count
    count+=1
    print(f'{count}: Saved partial results to {new_file_path}')

def simulate_chatgpt_api(title, snippet):
    """
    模擬對標題和片段進行處理的ChatGPT API。
    這裡僅返回隨機選擇的情感標籤作為範例。
    """
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    # 隨機選擇一個情感標籤
    return np.random.choice(sentiment_labels)


# 讀取CSV檔案，處理每行資料，並將結果保存到新的CSV檔案中。
process_csv_file(file_path)
# 讀取Excel檔案，處理每行資料，並將結果保存到新的CSV檔案中。
# process_excel_file(file_path)

# print (chat_with_openai('', 'high price'))
# print (chat_with_openai('', 'low price'))



