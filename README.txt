1.Python程式環境建立
conda create --name Stock_Predict python=3.10

2. 啟動Stock_Predict程式環境
activate Stock_Predict

2. 使用cd切換到你的程式目錄下，自行替換yourPath
cd yourPath

3.程式安裝相關套件
pip install -r requirements.txt

4.紐約時報抓取新聞程式執行
python NYT_article_search.py

5.ChatGPT API情緒分析程式執行
python ChatGPT_Sentiment_Request.py

6.計算每日平均情緒分數
python Sentiment_Score_Calculate.py

7.LSTM預測股價程式執行
python LSTM_Stock_Predict.py