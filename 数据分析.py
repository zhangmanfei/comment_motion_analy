# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:54:13 2020

@author: 29811
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置matplotlib绘图时的字体
my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")

# 读取数据
neg=pd.read_excel('D:/S/Learn/NLP/Jingdong/neg.xls',header=None,index=None)
pos=pd.read_excel('D:/S/Learn/NLP/Jingdong/pos.xls',header=None,index=None)
df=np.concatenate((pos[0], neg[0]))

#%%句子长度分布直方图
Num_len=[len(text) for text in df]
bins_interval=10#区间长度
bins=range(min(Num_len),max(Num_len)+bins_interval-1,bins_interval)#分组
plt.xlim(min(Num_len), max(Num_len))
plt.title("Probability-distribution")
plt.xlabel('Interval')
#plt.ylabel('Probability')
# 频率分布normed=True，频次分布normed=False
#prob,left,rectangle = plt.hist(x=Num_len, bins=bins, normed=True, histtype='bar', color=['r'])
plt.ylabel('Cumulative distribution')
prob,left,rectangle = plt.hist(x=Num_len, bins=bins,normed=True,cumulative=True, histtype='step', color=['r'])
plt.show()
#%%求分位点
import math

def quantile_p(data, p):
    data.sort()
    pos = (len(data) + 1)*p
    #pos = 1 + (len(data)-1)*p
    pos_integer = int(math.modf(pos)[1])
    pos_decimal = pos - pos_integer
    Q = data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1])*pos_decimal
    return Q

quantile=0.90#选取分位数
Q=quantile_p(Num_len,quantile)
print("\n分位点为%s的句子长度:%d." % (quantile, Q))
