# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:45:54 2023

@author: Ellen
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import jieba
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tqdm.pandas()
hotel_comments_ori = pd.read_csv('./booking_comments_分詞update.csv', 
              sep = ",", encoding = "UTF-8")
# 確認GPU是否可用
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 載入模型和tokenizer，並將它們移至正確的裝置上
model_name = "touch20032003/xuyuan-trial-sentiment-bert-chinese"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 情感類別和佔比的字典
id2label = {
    "0": "none",
    "1": "disgust",
    "2": "happiness",
    "3": "like",
    "4": "fear",
    "5": "sadness",
    "6": "anger",
    "7": "surprise"
}

df = hotel_comments_ori.groupby(['飯店名稱', '縣市', '鄉鎮']).filter(lambda x: x['飯店名稱'].count() > 30).reset_index(drop=True)

# 自訂情感分析函數
def sentiment_analysis_function(sentence):
    # 將評論分割成較小的片段
    max_length = 510
    segments = [sentence[i:i + max_length] for i in range(0, len(sentence), max_length)]
    print(segments)
    # 初始化情感結果
    result = {}
    for label in id2label.values():
        result[label] = 0.0

    # 對每個片段進行情感分析
    for segment in segments:
        tokens = tokenizer.encode_plus(segment, add_special_tokens=True, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_probs = torch.softmax(outputs.logits, dim=1).tolist()[0]

        # 將每個片段的情感結果加總
        for i, label in id2label.items():
            result[label] += predicted_probs[int(i)]

    # 計算平均值
    num_segments = len(segments)
    for label in id2label.values():
        result[label] /= num_segments
        result[label] = round(result[label], 2)
        
    print(result)    
    return result

#sentiment_analysis_function('太好了')

# 對每一則評論進行情感分析，並將結果加入 DataFrame 中
df['情緒分析'] = df['綜合評論'].apply(lambda x: sentiment_analysis_function(x))

# 初始化情緒平均字典
emotions = ['none', 'disgust', 'happiness', 'like', 'fear', 'sadness', 'anger', 'surprise']
hotel_emotion_averages = {}

# 分組計算每間飯店的情緒平均
#for hotel_name, group in df.groupby(['縣市','鄉鎮','飯店名稱']):
    #emotion_sum = {emotion: 0 for emotion in emotions}
    #num_reviews = len(group)

# 分組計算每間飯店的情緒平均
for (city, district, hotel), group in df.groupby(['縣市', '鄉鎮', '飯店名稱']):
    emotion_sum = {emotion: 0 for emotion in emotions}
    num_reviews = len(group)

    # 對每一則評論的情緒進行累加
    for emotion_dict in group['情緒分析']:
        for emotion, value in emotion_dict.items():
            emotion_sum[emotion] += value

    # 計算平均值
    emotion_avg = {emotion: round(emotion_sum[emotion] / num_reviews, 2) for emotion in emotions}
    if city not in hotel_emotion_averages:
        hotel_emotion_averages[city] = {}
    if district not in hotel_emotion_averages[city]:
        hotel_emotion_averages[city][district] = {}
    hotel_emotion_averages[city][district][hotel] = emotion_avg


import json
# 將 DataFrame 轉換為字典
#hotel_emotion_averages_dict = hotel_emotion_averages.to_dict()

    
# 寫入 JSON 檔案
with open('./json/booking_hotel_emotion_averages.json', 'w', encoding='utf_8_sig') as file:
    json.dump(hotel_emotion_averages, file, indent=4, ensure_ascii=False)

file_path = './booking_comments_分詞_情緒分析.csv'
df.to_csv(file_path, index=False, encoding='utf_8_sig')