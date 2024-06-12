import pandas as pd
import re
import glob
'''
將Google搜尋的新聞JSON格式轉換成CSV檔
'''

def extract_date(text):
    """使用正則表達式從文本中提取日期。"""
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if match:
        return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
    return None

# 定義處理JSON檔案的函數
def process_json_file(json_file_path):
    """處理單個 JSON 檔案並返回含日期的數據列表。"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = pd.read_json(file)
    
    rows = []
    for item in data.itertuples(index=False):
        # 遍歷文本列表
        for text in item.texts:
            date = extract_date(text)
            if date:
                # 移除文本中的逗號和單引號
                date_new = date.replace(",", "").replace("'", "")
                text1 = text.replace(",", "").replace("'", "")
                text2 = item.text.replace(",", "").replace("'", "")
                link_new = item.link.replace(",", "").replace("'", "")
                # 切掉 text1 的最後 10 個字元
                if len(text1) > 12:
                    text1 = text1[:-12]  # 去掉最後 10 個字元
                 # 切掉 text1 的最後 10 個字元
                if len(text2) > 12:
                    text2 = text2[:-12]  # 去掉最後 10 個字元
                row = {
                   'date': date_new + ',',
                    'text': text1 , # 確保沒有逗號
                    'texts': text2 ,  # 確保沒有逗號
                    'link': link_new # 假設連結沒有逗號
                }
                rows.append(row)
    return rows

# def process_json_file(json_file_path):
#     """處理單個 JSON 檔案並返回含日期的數據列表。"""
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         data = pd.read_json(file)
    
#     rows = []
#     for item in data.itertuples(index=False):
#         for text in item.texts:
#             date = extract_date(text)
#             if date:
#                 row = {
#                    'date': date,
#                     'text': item.text,
#                     'texts': text,
#                     'link': item.link
#                     # 'count': item.count,                  
#                 }
#                 rows.append(row)
#     return rows

def convert_jsons_to_excel(json_folder_path, excel_file_path):
    # 使用 glob 匹配資料夾內的所有 JSON 檔案
    json_files = glob.glob(f"{json_folder_path}/*.json")
    all_rows = []
    
    for json_file in json_files:
        file_rows = process_json_file(json_file)
        all_rows.extend(file_rows)
    
    # 如果沒有提取到任何含日期的數據，提前結束
    if not all_rows:
        print("沒有找到任何含日期的資料，不進行儲存。")
        return
    
    df = pd.DataFrame(all_rows)
    # 假設 df 是您的 DataFrame
    # 確保 'date' 欄位是日期格式
    df['date'] = pd.to_datetime(df['date'])

    # 按 'date' 欄位排序，由小到大
    df = df.sort_values(by='date')

    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    print(f"所有 JSON 檔案的數據已成功轉存為 Excel 檔案，儲存於：{excel_file_path}")

def convert_jsons_to_csv(json_folder_path, csv_file_path):
    # 使用 glob 匹配資料夾內的所有 JSON 檔案
    json_files = glob.glob(f"{json_folder_path}/*.json")
    all_rows = []
    
    for json_file in json_files:
        file_rows = process_json_file(json_file)
        all_rows.extend(file_rows)
        
    # 如果沒有提取到任何含日期的數據，提前結束
    if not all_rows:
        print("沒有找到任何含日期的資料，不進行儲存。")
        return
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"所有 JSON 檔案的數據已成功轉存為 CSV 檔案，儲存於：{csv_file_path}")

def convert_excel_to_csv(excel_file_path, csv_file_path, sheet_name=0):
    """
    將Excel檔案轉換成CSV檔案。
    
    參數:
    excel_file_path (str): Excel檔案的路徑。
    csv_file_path (str): 要存儲CSV檔案的路徑。
    sheet_name (str or int, optional): 要轉換的工作表名或索引。默認為0，即第一個工作表。
    
    返回:
    None
    """
    try:
        # 讀取指定的工作表
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        # 將DataFrame存儲為CSV檔案，不保存行索引
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"檔案已成功轉換並保存於：{csv_file_path}")
    except Exception as e:
        print(f"轉換過程中發生錯誤：{e}")

# 指定 JSON 資料夾和 Excel 檔案的路徑
json_folder_path = './Output_News/Amazon/2020.05.08-2020.09.11_Amazon_Google_News/'  # 替換為 JSON 檔案所在資料夾的實際路徑

# 存成Excel檔
# convert_jsons_to_excel(json_folder_path, '2020_Amazon_Google_News.xlsx')
# 存成CSV檔
convert_jsons_to_csv(json_folder_path, '2020_Amazon_Google_News.csv')

#將Excel檔案轉換成CSV檔案
# excel_file_path = '2020_Amazon_Google_News.xlsx'  # 替換成你的Excel檔案路徑
# csv_file_path = '2020_Amazon_Google_News.csv'  # 替換成你想要存儲CSV檔案的路徑
# convert_excel_to_csv(excel_file_path, csv_file_path, sheet_name='Sheet1')
print('完成!')