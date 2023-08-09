# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 09:37:46 2023

@author: Ellen
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rc('font', family='Microsoft JhengHei')

with open('./json/booking_hotel_emotion_averages.json', "r", encoding='utf_8_sig') as file:
    sentiment_data = json.load(file)
  
# 創建一個空的list來存放情緒"sadness"值
sadness_values = []

sentiment_data.keys()

# 遍歷JSON資料，找出情緒"sadness"值及其相關資訊
for county, town_data in sentiment_data.items():
    for town, hotel_data in town_data.items():
        for hotel, emotions in hotel_data.items():
            if 'sadness' in emotions:
                sadness_value = emotions['sadness']
                sadness_values.append((county, town, hotel, sadness_value))

# 將情緒"sadness"值由高至低排序
top_sadness_hotels = sorted(sadness_values, key=lambda x: x[3], reverse=True)[:5]

# 顯示前5名飯店及其相關資訊
for data in top_sadness_hotels:
    county, town, hotel, sadness = data
    print(f'縣市: {county}, 鄉鎮: {town}, 飯店名稱: {hotel}, 情緒"sadness"值: {sadness}')
    