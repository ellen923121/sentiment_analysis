# sentiment_analysis

1. sentiment_analyze.py
   - 先讀取飯店評論csv，再將每則綜合評論做情緒分析，最後將每間飯店所有評論情緒做平均，得出該飯店的情緒分析，並匯成json檔

3. emotion_radar.py
   - 將飯店的情緒分析json檔匯入，並畫成情緒雷達圖
     
 ![image](https://github.com/ellen923121/sentiment_analysis/blob/main/img/radar_example.png)

3. per_emotion_top_n.py
   - 篩選出某情緒最高或最低的飯店前n名 
