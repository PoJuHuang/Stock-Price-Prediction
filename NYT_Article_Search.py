from pynytimes import NYTAPI
import csv
import datetime
import os 
import time
# [自行設定]紐約時報API KEY
nyt = NYTAPI("GYrwuaJepShHfMX36ZUdFfdTmFq1GG7o", parse_dates=True)

# [自行設定]新聞搜尋關鍵字
# KEYWORD = "Tesla"
KEYWORD = "Amazon"
# KEYWORD = "Microsoft"

# 開始計時
start_time = time.time()
articles = nyt.article_search(
    query = KEYWORD, # Search for articles about Obama
    results = 20, # [自行設定]設定回傳幾篇結果，一次結果最多約2000篇左右
    # [自行設定]設定抓取新聞時間區間，Paper數據期間是2015年1月1日到2020年8月13日
    dates = {
        "begin": datetime.datetime(2020, 7, 13),
        "end": datetime.datetime(2020, 8, 11)
    },
    options = {
        "sort": "oldest", # Sort by oldest options
    }
)
def extract_article_info_v2(article):
    # 移除或替換字段內容中的逗號，防止破壞CSV格式
    title = article.get('headline', {}).get('main', '').replace(',', ';')
    snippet = article.get('snippet', '').replace(',', ';')
    pub_date = article.get('pub_date', '')
    byline = article.get('byline', {}).get('original', '')
    web_url = article.get('web_url', '')
    source = article.get('source', '')
    # 確保返回的是一個包含所有這些元素的元組
    return title, snippet, pub_date, byline, web_url, source

def extract_article_info(article):
    title = article.get('headline', {}).get('main', '')
    snippet = article.get('snippet', '')
    pub_date = article.get('pub_date', '')
    byline = article.get('byline', {}).get('original', '')
    # abstract = article.get('abstract', '')
    web_url = article.get('web_url', '')
    source = article.get('source', '')
    return title, snippet, pub_date, byline, web_url, source

def save_to_csv_ANSI(articles, filename):
    # 使用with語句打開檔案，並設置為ANSI編碼（cp1252）
    with open(filename, mode='w', newline='', encoding='cp1252', errors='replace') as file:
        writer = csv.writer(file)
        # 寫入標題行
        writer.writerow(['Title', 'Snippet', 'Published Date', 'Byline', 'Web_url', 'Source'])

        for article in articles:
            # 從每篇文章中提取信息
            title, snippet, pub_date, byline, web_url, source = extract_article_info_v2(article)
            # 確認提取的標題是否含有問題字符，如果有，則將其替換或處理
            title = title if '?' not in title else title.replace('?', '')
            # 將提取的信息寫入CSV文件
            writer.writerow([title, snippet, pub_date, byline, web_url, source])

def save_to_csv_UTF8(articles, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Snippet', 'Published Date', 'Byline',  'Web_url', 'Source'])

        for article in articles:
            title, snippet, pub_date, byline, web_url, source = extract_article_info_v2(article)
            # title後加","
            title = title + ","
            writer.writerow([title, snippet, pub_date, byline,  web_url, source])

# 獲取當前時間並格式化為字符串
# 例如：'2024_01_30_15_30_00' 格式
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# 創建一個名為 'Output' 的新資料夾，如果它不存在的話
output_dir = 'Output_News'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 構建檔案名稱，包括路徑
filename = os.path.join(output_dir, f"{current_time}_{KEYWORD}_News_Results.csv")

# 將新聞存儲為 CSV 檔案
save_to_csv_UTF8(articles, filename)
print(f"News data saved to {filename}")
# 結束計時
end_time = time.time()
print(f"程式執行時間：{end_time - start_time}秒")
print(f"程式執行時間：{(end_time - start_time)/60}分")
