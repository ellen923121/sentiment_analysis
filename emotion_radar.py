# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:30:56 2023

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
  

def sentiment_radar_chart(city, town, hotel_name):
    
    #設定emotions順序
    emotions = ["none","like", "happiness", "surprise","fear","disgust","sadness","anger"]
    values=[]
    for emo in emotions:
        values.append(sentiment_data[city][town][hotel_name][emo])
    
    # 將圓形分割成八等份
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
    
    # 使圖形封閉（連接第一個點和最後一個點）
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # 加上第一個情緒的值
    emotions.append(emotions[0])
    
    plt.figure(figsize=(800,600))
    # 設置極座標圖的參數
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # 設置刻度定位器和格式化器
    ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0.1, 1.0, 0.1)))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # 設置極座標軸標籤
    ax.set_thetagrids(angles * 180 / np.pi, emotions, va="center", fontsize = 18)
    
     # 設置標題
    plt.title(f'{hotel_name} 情緒雷達圖', fontsize = 24)


    # 調整圖和子圖之間的間距，增加top和bottom參數的值以增加間距
    plt.subplots_adjust(top=0.85, bottom=0.1)
    # 顯示圖形
    plt.show()

    
if __name__=='__main__':
    sentiment_radar_chart("宜蘭縣", "羅東鎮", "立琦大飯店 ")