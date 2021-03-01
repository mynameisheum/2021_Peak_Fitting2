#!/usr/bin/env python
# coding: utf-8

# 

# In[285]:



import numpy as np

# peak_0_data_number = 100000
peak_1_data_number = 300000
peak_2_data_number = 300000
peak_3_data_number = 300000


peak_1_graph1_param = np.zeros((peak_1_data_number,3))

peak_2_graph1_param = np.zeros((peak_2_data_number,3))
peak_2_graph2_param = np.zeros((peak_2_data_number,3))

peak_3_graph1_param = np.zeros((peak_3_data_number,3))
peak_3_graph2_param = np.zeros((peak_3_data_number,3))
peak_3_graph3_param = np.zeros((peak_3_data_number,3))


# In[286]:


import keras 
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend


# In[287]:


print(keras.__version__)
print(tf.__version__)


# In[288]:


print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


# In[ ]:





# In[ ]:





# In[289]:


# x축 범위,격자 설정 ( data의 크기 설정)

x = np.linspace(0,15,401)


# In[290]:


# voight function 설정

def y(a,b,c,x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7*np.exp(-np.log(2) * (x - a)**2 / (beta * b)**2)) + (0.3 / (1 + (x -a)**2 / (gamma * b)**2)))
    return y


# In[291]:


# data의 parameter 만드는 작업
# [center, width, amp]



for i in range(peak_1_data_number):
    peak_1_graph1_param[i] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]
    
for j in range(peak_2_data_number):
    peak_2_graph1_param[j] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]
    peak_2_graph2_param[j] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]

for k in range(peak_3_data_number):
    peak_3_graph1_param[k] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]
    peak_3_graph2_param[k] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]
    peak_3_graph3_param[k] = [2+ 11.0*np.random.rand(), 0.3 + 1.6*np.random.rand(), 0.05 + np.random.rand()]
    
    


# In[292]:


# print(peak_0_graph0_param.shape)
print(peak_1_graph1_param.shape)
print(peak_2_graph1_param.shape)
print(peak_3_graph1_param.shape)


# In[293]:


import matplotlib.pyplot as plt

plt.figure(figsize = (5,10))
plt.subplot(3,1,1)
plt.plot(x,y(peak_1_graph1_param[0][0],peak_1_graph1_param[0][1],peak_1_graph1_param[0][2],x), c = 'black')
plt.ylim(0,3)
plt.title('each peak of peak1 graph ')

plt.subplot(3,1,2)
plt.plot(x,y(peak_2_graph1_param[0][0],peak_2_graph1_param[0][1],peak_2_graph1_param[0][2],x), c = 'blue')
plt.plot(x,y(peak_2_graph2_param[0][0],peak_2_graph2_param[0][1],peak_2_graph2_param[0][2],x), c = 'blue')
plt.ylim(0,3)
plt.title('each peak of peak2 graph')

plt.subplot(3,1,3)
plt.plot(x,y(peak_3_graph1_param[0][0],peak_3_graph1_param[0][1],peak_3_graph1_param[0][2],x), c = 'green')
plt.plot(x,y(peak_3_graph2_param[0][0],peak_3_graph2_param[0][1],peak_3_graph2_param[0][2],x), c = 'green')
plt.plot(x,y(peak_3_graph3_param[0][0],peak_3_graph3_param[0][1],peak_3_graph3_param[0][2],x), c = 'green')
plt.ylim(0,3)
plt.title('each peak of peak3 graph')


# In[294]:


# noise 생성

noise_level = 0.05

peak_1_graph1_noise=[]
for i in range(peak_1_data_number):
    peak1_noise = []
    for a in range(len(x)):
        peak1_noise.append(np.random.rand() * noise_level - noise_level * 0.5)
    peak_1_graph1_noise.append(peak1_noise)

# peak_2_graph1_noise = []
peak_2_graph2_noise = []
for j in range(peak_2_data_number):
#     peak2_noise1 = []
    peak2_noise2 = []
    for b in range(len(x)):
#         peak2_noise1.append(np.random.rand() * noise_level - noise_level * 0.5)
        peak2_noise2.append(np.random.rand() * noise_level - noise_level * 0.5)
#     peak_2_graph1_noise.append(peak2_noise1)
    peak_2_graph2_noise.append(peak2_noise2)
    
    
# peak_3_graph1_noise = []
# peak_3_graph2_noise = []
peak_3_graph3_noise = []
for k in range(peak_3_data_number):
#     peak3_noise1 = []
#     peak3_noise2 = []
    peak3_noise3 = []
    for c in range(len(x)):
#         peak3_noise1.append(np.random.rand() * noise_level - noise_level * 0.5)
#         peak3_noise2.append(np.random.rand() * noise_level - noise_level * 0.5)
        peak3_noise3.append(np.random.rand() * noise_level - noise_level * 0.5)
#     peak_3_graph1_noise.append(peak3_noise1)
#     peak_3_graph2_noise.append(peak3_noise2)
    peak_3_graph3_noise.append(peak3_noise3)


# In[295]:



# peak_0_graph0 = np.zeros((100000,401))
# peak_0_graph0_param.shape


# In[296]:



# peak_0_data_number = 100000

# peak0_param = np.zeros((peak_0_data_number,4))
# peak0 = np.zeros((100000,401))


# In[297]:


# for i in range(peak_0_data_number):
#     for j in range(401):
#         peak0[i][j] = (np.random.rand() * noise_level - noise_level * 0.5)


# In[298]:


# peak0.shape
# peak0_param.shape


# In[299]:


# plt.plot(peak0[-1])
# plt.ylim(0,1)


# In[300]:


# 생성한 noise 넣어주기

peak_1_graph1 = []
for i in range(peak_1_data_number):
    peak_1_graph1.append(y(peak_1_graph1_param[i][0],peak_1_graph1_param[i][1],peak_1_graph1_param[i][2],x) + np.array(peak_1_graph1_noise[i]))

peak_2_graph1 = []
peak_2_graph2 = []
for j in range(peak_2_data_number):
    peak_2_graph1.append(y(peak_2_graph1_param[j][0],peak_2_graph1_param[j][1],peak_2_graph1_param[j][2],x) )
    peak_2_graph2.append(y(peak_2_graph2_param[j][0],peak_2_graph2_param[j][1],peak_2_graph2_param[j][2],x) + np.array(peak_2_graph2_noise[k]))

peak_3_graph1 = []
peak_3_graph2 = []
peak_3_graph3 = []
for k in range(peak_3_data_number):
    peak_3_graph1.append(y(peak_3_graph1_param[k][0],peak_3_graph1_param[k][1],peak_3_graph1_param[k][2],x) )
    peak_3_graph2.append(y(peak_3_graph2_param[k][0],peak_3_graph2_param[k][1],peak_3_graph2_param[k][2],x) )
    peak_3_graph3.append(y(peak_3_graph3_param[k][0],peak_3_graph3_param[k][1],peak_3_graph3_param[k][2],x) + np.array(peak_3_graph3_noise[k]))


# In[301]:


len(peak_3_graph2)
len(peak_3_graph1_param+peak_3_graph2_param+peak_3_graph3_param)


# In[302]:


# noise 확인

import matplotlib.pyplot as plt

plt.figure(figsize = (5,10))
plt.subplot(3,1,1)
plt.ylim(0,3)
plt.plot(x,peak_1_graph1[0],c= 'black')
plt.title('each peak of peak1 graph with noise')

plt.subplot(3,1,2)
plt.ylim(0,3)
plt.plot(x,peak_2_graph1[0], c = 'blue')
plt.plot(x,peak_2_graph2[0], c = 'blue')
plt.title('each peak of peak2 graph with noise')

plt.subplot(3,1,3)
plt.ylim(0,3)
plt.plot(x, peak_3_graph1[0], c = 'green')
plt.plot(x, peak_3_graph2[0], c = 'green')
plt.plot(x, peak_3_graph3[0], c = 'green')
plt.title('each peak of peak3 graph with noise')


# In[303]:


peak1 = peak_1_graph1

peak2 = []
for i in range(peak_2_data_number):
    peak2.append(np.array(peak_2_graph1[i]) + np.array(peak_2_graph2[i]))

peak3 = []
for j in range(peak_3_data_number):
    peak3.append(np.array(peak_3_graph1[j])+
                np.array(peak_3_graph2[j])+
                np.array(peak_3_graph3[j]))
    


# In[304]:


peak2[0]


# In[305]:


# 각 peak1,2,3의 그래프 결과

plt.figure(figsize=(5,10))
plt.subplot(3,1,1)
plt.ylim(0,3)
plt.plot(x,peak1[0], c = 'black')
plt.title('peak1')

plt.figure(figsize=(5,10))
plt.subplot(3,1,2)
plt.ylim(0,3)
plt.plot(x,peak2[0], c = 'blue')
plt.title('peak2')

plt.figure(figsize=(5,10))
plt.subplot(3,1,3)
plt.ylim(0,3)
plt.plot(x,peak3[0], c = 'green')
plt.title('peak3')


# In[306]:


# 뭉특한걸 빼는 거니까 amp를 가장 높은걸로 parameter를 data에 labeling 해주자-> center로 수정


# In[307]:


# peak1 data의 labeling 작업 ( 여기서 peak_number까지 넣어주자)

peak1_param = []
for i in range(peak_1_data_number):
    peak1_param.append(list(np.append(peak_1_graph1_param[i], np.array(1))))

# peak_3_graph1_param
# peak_3_graph2_param
# peak_3_graph3_param


# In[308]:


# label 확인 (list형식, 4자리수의 파라미터)

peak1_param[0]


# In[309]:


len(peak_2_graph1[0])


# In[310]:


# peak2 data의 labeling 작업 ( 여기서 peak_number까지 넣어주자)
# center가 높은걸로

peak2_param = []
for i in range(peak_2_data_number):
    
#     if peak_2_graph1_param[i][0]> peak_2_graph2_param[i][0]:
#         peak2_param.append(list(np.append(peak_2_graph1_param[i], np.array(2))))
#     elif peak_2_graph1_param[i][0]< peak_2_graph2_param[i][0]:
#         peak2_param.append(list(np.append(peak_2_graph2_param[i], np.array(2))))

    if sum(peak_2_graph1[i])> sum(peak_2_graph2[i]):
        peak2_param.append(list(np.append(peak_2_graph1_param[i], np.array(2))))
    elif sum(peak_2_graph1[i])< sum(peak_2_graph2[i]):
        peak2_param.append(list(np.append(peak_2_graph2_param[i], np.array(2))))


# In[311]:


# label 확인 (list형식, 4자리수의 파라미터)

peak2_param[0]


# In[312]:


# peak2개 일때 center높은걸로 라벨을 지정해줬는지 그래프로 확인

for i in range(10):
    plt.figure(figsize = (10,5))
    plt.ylim(0,3)
    plt.plot(x,y(peak2_param[i][0],peak2_param[i][1],peak2_param[i][2],x),c = 'red')
    plt.plot(x,peak_2_graph1[i], c = 'blue')
    plt.plot(x,peak_2_graph2[i], c = 'blue')
    plt.plot(x,peak_2_graph1[i]+peak_2_graph2[i], c = 'black')
#     plt.plot(x,peak2[i], c = 'green')


# In[313]:


print(peak_2_graph1_param[5])
print(peak_2_graph2_param[5])


# In[314]:


# peak3 data의 labeling 작업 ( 여기서 peak_number까지 넣어주자)
# center값이 높은걸로

peak3_param = []

for i in range(peak_3_data_number):
    
#     if peak_3_graph1_param[i][0]> peak_3_graph2_param[i][0] and peak_3_graph1_param[i][0] >peak_3_graph3_param[i][0]:
#         peak3_param.append(list(np.append(peak_3_graph1_param[i], np.array(3))))
    
#     elif peak_3_graph2_param[i][0]> peak_3_graph1_param[i][0] and peak_3_graph2_param[i][0] >peak_3_graph3_param[i][0]:
#         peak3_param.append(list(np.append(peak_3_graph2_param[i], np.array(3))))
 
#     elif peak_3_graph3_param[i][0]> peak_3_graph2_param[i][0] and peak_3_graph3_param[i][0] >peak_3_graph1_param[i][0]:
#         peak3_param.append(list(np.append(peak_3_graph3_param[i], np.array(3))))
        
        
    if sum(peak_3_graph1[i])> sum(peak_3_graph2[i]) and sum(peak_3_graph1[i]) >sum(peak_3_graph3[i]):
        peak3_param.append(list(np.append(peak_3_graph1_param[i], np.array(3))))
    
    elif sum(peak_3_graph2[i])> sum(peak_3_graph1[i]) and sum(peak_3_graph2[i]) >sum(peak_3_graph3[i]):
        peak3_param.append(list(np.append(peak_3_graph2_param[i], np.array(3))))
 
    elif sum(peak_3_graph3[i])> sum(peak_3_graph2[i]) and sum(peak_3_graph3[i]) >sum(peak_3_graph1[i]):
        peak3_param.append(list(np.append(peak_3_graph3_param[i], np.array(3))))
        
        


# In[315]:


# 3개의 peak들이 비교가 잘 되었는지 확인하기

i = 8
print(peak_3_graph3_param[i][2]> peak_3_graph2_param[i][2] and peak_3_graph3_param[i][2] >peak_3_graph1_param[i][2])
print(peak_3_graph2_param[i][2]> peak_3_graph1_param[i][2] and peak_3_graph2_param[i][2] >peak_3_graph3_param[i][2])
print(peak_3_graph1_param[i][2]> peak_3_graph2_param[i][2] and peak_3_graph1_param[i][2] >peak_3_graph3_param[i][2])


# In[316]:


# peak3개일때 center높은걸로 라벨을 지정해줬는지 그래프로 확인

for i in range(50):
    plt.figure(figsize = (10,5))
    plt.ylim(0,3)
    plt.plot(x,y(peak3_param[i][0],peak3_param[i][1],peak3_param[i][2],x),c = 'red')
    plt.plot(x,peak_3_graph1[i], c = 'blue')
    plt.plot(x,peak_3_graph2[i], c = 'blue')
    plt.plot(x,peak_3_graph3[i], c = 'blue')
    plt.plot(x,peak_3_graph1[i]+peak_3_graph2[i]+peak_3_graph3[i], c = 'black')


# In[317]:


peak3_param[5]


# In[318]:


# ---------------------------------
# h = peak_1_graph1_param[0].copy()
# # h.append([0])
# np.append(h, np.array(1))
# # h
# ---------------------------------


# In[319]:


# arr = np.array([])
# arr = np.append(arr, np.array([1, 2, 3]))
# arr = np.append(arr, np.array([4, 5]))
# arr


# In[320]:


# print(np.array(peak0).shape)
print(np.array(peak1).shape)
print(np.array(peak2).shape)
print(np.array(peak3).shape)

# print(np.array(peak0_param).shape)
print(np.array(peak1_param).shape)
print(np.array(peak2_param).shape)
print(np.array(peak3_param).shape)


# In[321]:


# peak00_param = []
# for i in range(len(peak0)):
#     peak00_param.append(list(peak0_param[i]))


# In[322]:


# peak00 = []
# for i in range(len(peak0)):
    
#     peak00.append(peak0[i])


# In[323]:


# shuffle해주기

# len(peak0+peak1+peak2+peak3)
# len(peak0_param + peak1_param + peak2_param + peak3_param)

# peak = peak00 + peak1+ peak2 + peak3
# peak_param = peak00_param + peak1_param + peak2_param + peak3_param

peak = peak1+ peak2 + peak3
peak_param = peak1_param + peak2_param + peak3_param


# In[324]:


# peak1[0]


# In[325]:


len(peak1+peak2+peak3)


# In[326]:


before_shuffle = []
for i in zip(peak, peak_param):
    before_shuffle.append(i)
    
after_shuffle = before_shuffle


# In[327]:


len(peak_3_graph1)
len(peak)


# In[328]:


# -------------------
# t = [1,2,3]
# t2 = [11,12,13]
# t3 = [21,22,23]
# for i in zip(t,t2,t3):
#     print(i)
# -------------------


# In[329]:


import random

random.shuffle(after_shuffle)


# In[330]:


after_shuffle_peak = []
after_shuffle_peak_param = []

for i in range(len(after_shuffle)):
    after_shuffle_peak.append(after_shuffle[i][0])
    after_shuffle_peak_param.append(after_shuffle[i][1])
    


# In[331]:


# 셔플 순서대로 잘 됐는지 확은해주기

x = np.linspace(0,15,401)
# x = np.linspace(0,7,401)
for i in range(40):
    
    plt.figure(figsize = (10,5))
    plt.plot(x,after_shuffle_peak[i], c = 'black')
    plt.plot(x,y(after_shuffle_peak_param[i][0],after_shuffle_peak_param[i][1],after_shuffle_peak_param[i][2],x), c = 'blue')
    plt.ylim(0,1)


# In[332]:


print(after_shuffle_peak_param[2])


# In[333]:


print(peak[0])
print(after_shuffle_peak[0])


# In[334]:


# 함수형 api를 위해 paramter를 재분배


center = []
width = []
amp = []
peak_number = []

for i in range(len(after_shuffle)):
    center.append(after_shuffle_peak_param[i][0])
    width.append(after_shuffle_peak_param[i][1])
    amp.append(after_shuffle_peak_param[i][2])
    peak_number.append(after_shuffle_peak_param[i][3])


# In[335]:


len(after_shuffle)


# In[55]:


# import pandas as pd

# df_peak1_param = pd.DataFrame(peak1_param)
# df_peak2_param = pd.DataFrame(peak2_param)
# df_peak3_param = pd.DataFrame(peak3_param)
# df_center = pd.DataFrame(center)
# df_width = pd.DataFrame(width)
# df_amp = pd.DataFrame(amp)
# df_after_shuffle_peak= pd.DataFrame(after_shuffle_peak) 
# df_after_shuffle_peak_param= pd.DataFrame(after_shuffle_peak_param) 


# df_peak1_param.to_csv('second project peak1 param_30')
# df_peak2_param.to_csv('second project peak2 param_30')
# df_peak3_param.to_csv('second project peak3 param_30')
# df_center.to_csv('second project center_90')
# df_width.to_csv('second project width_90')
# df_amp.to_csv('second project amp_90')
# df_after_shuffle_peak.to_csv('second project after suffle peak')
# df_after_shuffle_peak_param.to_csv('second project after suffle peak param')


# In[1]:


import pandas as pd
import numpy as np


# In[235]:



# graph = pd.read_csv('graph_1000.csv').values[:,1:]
# center= pd.read_csv('second project center_90').values[:,1:]
# width= pd.read_csv('second project width_90').values[:,1:]
# amp= pd.read_csv('second project amp_90').values[:,1:]

# after_shuffle_peak = pd.read_csv('second project after suffle peak').values[:,1:]
# after_shuffle_peak_param = pd.read_csv('second project after suffle peak param').values[:,1:]


# In[338]:


# after_shuffle_peak.shape
# after_shuffle_peak_param.shape
after_shuffle_peak_param[2]


# In[339]:


len(after_shuffle_peak)


# In[340]:


center = []
width = []
amp = []
peak_number = []

for i in range(len(after_shuffle_peak)):
    center.append(after_shuffle_peak_param[i][0])
    width.append(after_shuffle_peak_param[i][1])
    amp.append(after_shuffle_peak_param[i][2])
    peak_number.append(after_shuffle_peak_param[i][3])


# In[341]:


# voight function 설정

def y(a,b,c,x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7*np.exp(-np.log(2) * (x - a)**2 / (beta * b)**2)) + (0.3 / (1 + (x -a)**2 / (gamma * b)**2)))
    return y


# In[342]:


import matplotlib.pyplot as plt

x = np.linspace(0,15,401)
# x = np.linspace(0,7,401)
for i in range(40):
    
    plt.figure(figsize = (10,5))
    plt.plot(x,after_shuffle_peak[i], c = 'black')
    plt.plot(x,y(after_shuffle_peak_param[i][0],after_shuffle_peak_param[i][1],after_shuffle_peak_param[i][2],x), c = 'blue')
    plt.ylim(0,1)


# In[ ]:





# In[343]:


# 잘 재분배 됐는지 peak_number로 확인

for i in range(30):
    print('peak_number : ',peak_number[i], 'after_shuffle_peak_parma : ',after_shuffle_peak_param[i][3])


# In[344]:


print(len(after_shuffle_peak))
print(len(center))
print(len(width))
print(len(amp))
print(len(peak_number))
0.8*len(center)


# In[345]:


len(after_shuffle_peak[0])


# In[346]:


# train : val : test => 8: 1: 1로 나누기

train_peak = after_shuffle_peak[:int(0.8*len(after_shuffle_peak))]
val_peak = after_shuffle_peak[int(0.8*len(after_shuffle_peak)):int(0.9*len(after_shuffle_peak))]
test_peak = after_shuffle_peak[int(0.9*len(after_shuffle_peak)):]

train_center = center[:int(0.8*len(center))]
val_center = center[int(0.8*len(center)):int(0.9*len(center))]
test_center = center[int(0.9*len(center)):]


train_width = width[:int(0.8*len(width))]
val_width = width[int(0.8*len(width)):int(0.9*len(width))]
test_width = width[int(0.9*len(width)):]

train_amp = amp[:int(0.8*len(amp))]
val_amp = amp[int(0.8*len(amp)):int(0.9*len(amp))]
test_amp = amp[int(0.9*len(amp)):]

train_peak_number = peak_number[:int(0.8*len(peak_number))]
val_peak_number = peak_number[int(0.8*len(peak_number)):int(0.9*len(peak_number))]
test_peak_number = peak_number[int(0.9*len(peak_number)):]


# In[347]:


# 8: 1: 1로 나누어졌는지 확인

print(len(train_peak))
print(len(val_peak))
print(len(test_peak))
print('\n')

print(len(train_center))
print(len(val_center))
print(len(test_center))
print('\n')

print(len(train_width))
print(len(val_width))
print(len(test_width))
print('\n')

print(len(train_amp))
print(len(val_amp))
print(len(test_amp))
print('\n')

print(len(train_peak_number))
print(len(val_peak_number))
print(len(test_peak_number))


# In[348]:


np.array(train_peak)[0][0]


# In[349]:


# conv1d를 위해 reshape로 1차원 늘리기

train_peak = np.array(train_peak).reshape(np.array(train_peak).shape[0],np.array(train_peak).shape[1],1)
val_peak = np.array(val_peak).reshape(np.array(val_peak).shape[0],np.array(val_peak).shape[1],1)
test_peak = np.array(test_peak).reshape(np.array(test_peak).shape[0],np.array(test_peak).shape[1],1)

print(train_peak.shape)
print(val_peak.shape)
print(test_peak.shape)


# In[350]:


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Add,advanced_activations
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers.merge import Concatenate
from keras import layers


# In[351]:


# se-resnet 기존 한번
# se-resnet add() 후 leakyrelu없앤거 한번
# se_res_densenet 학습
# se_resx_densenet 학습


# In[352]:


from keras.layers import Activation, Multiply, Reshape


# In[21]:


1+1


# In[27]:


################### SE- resx- Densenet 2


# In[22]:



x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
shortcut = 0
se = 0
Cf = 0.5

# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2
# 443
# --------------------------------------
shortcut_1_4 = x
shortcut_1_3 = x
shortcut_1_2 = x
shortcut_1_1 = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

# 여기다 conv1 넣고 chennel을 반으로 줄이고
first_layer = []
for i in range(int(64/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    first_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer)

# 다시 conv1d로 2배로 펌핑하고
x = layers.Conv1D(64,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

# 
se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(64 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(64, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut]) # x.shape = (100,256)

shortcut = x


first_layer2 = []
for i in range(int(64/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    first_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer2)

x = layers.Conv1D(64,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)


se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(64, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,100])(se)

x = Multiply()([x,se])


x = layers.Add()([x,shortcut])
x = Concatenate()([x,shortcut_1_1])

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(64 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# --------------------------------------
shortcut_2_4 = x
shortcut_2_3 = x
shortcut_2_2 = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


second_layer = []
for i in range(int(128/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer)

x = layers.Conv1D(128,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(128 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(128, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x

second_layer2 = []
for i in range(int(128/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer2)

x = layers.Conv1D(128,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(128, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se])

# transition layer---------------------------------
x = layers.Add()([x,shortcut])
shortcut_1_2 = layers.BatchNormalization()(shortcut_1_2)
shortcut_1_2 = layers.LeakyReLU(alpha = 0.01)(shortcut_1_2)
shortcut_1_2 = layers.Conv1D(64,1,  padding = 'same', kernel_initializer='he_normal')(shortcut_1_2)
shortcut_1_2 = AveragePooling1D(2, padding ='same')(shortcut_1_2)

x = Concatenate()([x,shortcut_1_2])
x = Concatenate()([x,shortcut_2_2])

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(128 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# # --------------------------------------
shortcut_3_4 = x
shortcut_3_3 = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


third_layer = []
for i in range(int(256/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer)

x = layers.Conv1D(256,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(256 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(256, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x

third_layer2 = []
for i in range(int(256/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer2)

x = layers.Conv1D(256,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(256, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

# transition layer---------------------------------

shortcut_1_3 = layers.BatchNormalization()(shortcut_1_3)
shortcut_1_3 = layers.LeakyReLU(alpha = 0.01)(shortcut_1_3)
shortcut_1_3 = layers.Conv1D(64,1,strides = 2,  padding = 'same', kernel_initializer='he_normal')(shortcut_1_3)
shortcut_1_3 = AveragePooling1D(2, padding ='same')(shortcut_1_3)

x = Concatenate()([x,shortcut_1_3])

shortcut_2_3 = layers.BatchNormalization()(shortcut_2_3)
shortcut_2_3 = layers.LeakyReLU(alpha = 0.01)(shortcut_2_3)
shortcut_2_3 = layers.Conv1D(64,1,strides = 1,  padding = 'same', kernel_initializer='he_normal')(shortcut_2_3)
shortcut_2_3 = AveragePooling1D(2, padding ='same')(shortcut_2_3)

x = Concatenate()([x,shortcut_2_3])
x = Concatenate()([x,shortcut_3_3])

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(256 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# # --------------------------------------
shortcut_4_4 = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


four_layer = []
for i in range(int(512/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer)

x = layers.Conv1D(512,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(512 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(512, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x

four_layer2 = []
for i in range(int(512/32)):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer2)

x = layers.Conv1D(512,1,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(512, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

shortcut_1_4 = layers.BatchNormalization()(shortcut_1_4)
shortcut_1_4 = layers.LeakyReLU(alpha = 0.01)(shortcut_1_4)
shortcut_1_4 = layers.Conv1D(64,1,strides = 2,  padding = 'same', kernel_initializer='he_normal')(shortcut_1_4)
shortcut_1_4 = layers.Conv1D(64,1,strides = 2,  padding = 'same', kernel_initializer='he_normal')(shortcut_1_4)
shortcut_1_4 = AveragePooling1D(2, padding ='same')(shortcut_1_4)

x = Concatenate()([x,shortcut_1_4])

shortcut_2_4 = layers.BatchNormalization()(shortcut_2_4)
shortcut_2_4 = layers.LeakyReLU(alpha = 0.01)(shortcut_2_4)
shortcut_2_4 = layers.Conv1D(64,1,strides = 2,  padding = 'same', kernel_initializer='he_normal')(shortcut_2_4)
shortcut_2_4 = AveragePooling1D(2, padding ='same')(shortcut_2_4)

x = Concatenate()([x,shortcut_2_4])

shortcut_3_4 = layers.BatchNormalization()(shortcut_3_4)
shortcut_3_4 = layers.LeakyReLU(alpha = 0.01)(shortcut_3_4)
shortcut_3_4 = layers.Conv1D(64,1,strides = 1,  padding = 'same', kernel_initializer='he_normal')(shortcut_3_4)
shortcut_3_4 = AveragePooling1D(2, padding ='same')(shortcut_3_4)

x = Concatenate()([x,shortcut_3_4])
x = Concatenate()([x,shortcut_4_4])
x= layers.LeakyReLU(alpha = 0.01)(x)


# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)


model_se_resx_densenet2 = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3])
print(model_se_resx_densenet2.summary())


# In[23]:


plot_model(model_se_resx_densenet2,show_shapes = True)


# In[24]:


model_se_resx_densenet2.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse'},
#                       'Dense3_peak_number' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91},
#                             'Dense3_peak_number' :0.33 },
              metrics = ['mae'])


# In[25]:


#  콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_se_resx_densenet2.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[26]:


model_se_resx_densenet2=model_se_resx_densenet2.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[27]:


for key in model_se_resx_densenet2.history.keys():
    print(key)
  


# In[29]:



plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(model_se_resx_densenet2.history['loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_resx_densenet2.history['val_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(model_se_resx_densenet2.history['total_center3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_resx_densenet2.history['val_total_center3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()

plt.subplot(234)
plt.plot(model_se_resx_densenet2.history['total_width3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_resx_densenet2.history['val_total_width3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(model_se_resx_densenet2.history['total_amp3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_resx_densenet2.history['val_total_amp3_loss'], 'r:', label = 'light -SE-Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[30]:


# model_se_resx_densenet2

from tensorflow.keras.models import load_model

best_model_se_resx_densenet2 = load_model('best_model_se_resx_densenet2.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_resx_densenet2.summary()


# In[31]:



prediction_se_resx_densenet2 = best_model_se_resx_densenet2.predict(test_peak)
print(len(prediction_se_resx_densenet2))
print(np.array(prediction_se_resx_densenet2).shape)


# In[32]:


loss_center = 0
loss_width = 0
loss_amp = 0
# loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_resx_densenet2[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_resx_densenet2[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_resx_densenet2[2][i][0]))
#     loss_peak_number += abs((test_peak_number[i] - prediction[0][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
# loss_peak_number = loss_peak_number/len(test_center)


# In[33]:


print(loss_center)
print(loss_width)
print(loss_amp)


# In[ ]:





# In[ ]:


# 여기다 새로운 se-resenex-dense-inception parameter 맞춘


# In[251]:



x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
shortcut = 0
se = 0
Cf = 0.5
cardinality = 16

# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2
# 443
# --------------------------------------
# shortcut_1_4 = x
# shortcut_1_3 = x
# shortcut_1_2 = x
# shortcut_1_1 = x

shortcut_dense = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

# ------------------ first layer-1
# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

first_layer = []
for i in range(cardinality):

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)   # residual connection이 아니니까 activation 활성화시켜보자
    first_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer)
x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)


# 
se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(64 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(64, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut]) # x.shape = (100,256)


# ----------------- first layer-2

shortcut = x


# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

first_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    first_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer2)
x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)



se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(64, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,100])(se)

x = Multiply()([x,se])


x = layers.Add()([x,shortcut])

# -----------------------------transition layer


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x = layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(64 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# ----------------- second layer-1

# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

# x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

second_layer = []
for i in range(cardinality):

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer)
x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(128 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(128, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])


# ----------------- second layer-2

shortcut = x

# x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

second_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer2)
x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(128, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se])
x = layers.Add()([x,shortcut])

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(128 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# ----------------- third layer-1


# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

# x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

third_layer = []
for i in range(cardinality):

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer)
x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(256 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(256, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

# ----------------- third layer-2

shortcut = x



# x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

third_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer2)
x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(256, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(256 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# --------------------------------------
# ----------------- four layer-1


# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


# x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

four_layer = []
for i in range(cardinality):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer)
x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(512 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(512, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

# ----------------- four layer-2

shortcut = x



# x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x) # 786,994

four_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer2)
x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x) 

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(512, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

x = Concatenate()([x,shortcut_dense])

# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)


model_se_res_dense_inceptionnet2 = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_se_res_dense_inceptionnet2.summary())


# In[252]:


plot_model(model_se_res_dense_inceptionnet2,show_shapes = True)


# In[256]:


model_se_res_dense_inceptionnet2.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[ ]:


#  콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_se_res_dense_inceptionnet2.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[ ]:


model_se_res_dense_inceptionnet2=model_se_res_dense_inceptionnet2.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp),np.array(val_peak_number)]),
            callbacks = [ early_stopping,model_checkpoint, reduce_lr],
                                                                     shuffle = True)


# In[ ]:


for key in model_se_res_dense_inceptionnet2.history.keys():
    print(key)
  


# In[ ]:



plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(model_se_res_dense_inceptionnet2.history['loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet2.history['val_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(model_se_res_dense_inceptionnet2.history['total_center3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet2.history['val_total_center3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()
plt.ylim(0,1)

plt.subplot(234)
plt.plot(model_se_res_dense_inceptionnet2.history['total_width3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet2.history['val_total_width3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(model_se_res_dense_inceptionnet2.history['total_amp3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet2.history['val_total_amp3_loss'], 'r:', label = 'light -SE-Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[ ]:


plt.plot(model_se_res_dense_inceptionnet2.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet2.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.03)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[ ]:



from tensorflow.keras.models import load_model

best_model_se_res_dense_inceptionnet2 = load_model('best_model_se_res_dense_inceptionnet2.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_res_dense_inceptionnet2.summary()


# In[ ]:



prediction_se_res_dense_inceptionnet2 = best_model_se_res_dense_inceptionnet2.predict(test_peak)
print(len(prediction_se_res_dense_inceptionnet2))
print(np.array(prediction_se_res_dense_inceptionnet2).shape)


# In[ ]:


x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_se_res_dense_inceptionnet2[0][i][0],prediction_se_res_dense_inceptionnet2[1][i][0],prediction_se_res_dense_inceptionnet2[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[ ]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_res_dense_inceptionnet2[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_res_dense_inceptionnet2[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_res_dense_inceptionnet2[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_se_res_dense_inceptionnet2[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[ ]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)


# In[ ]:





# In[ ]:





# In[ ]:


# new densenet
# explain

# 파라미터는 5,000,000미만으로

# Densnet
# densene의 channel의 reuse의 concept를 가져와서 concatenate을 통해 block마다 connection을 해줌
# transition layer의 Compositon factor( = 0.5) concept으로 first block->second block->..foour block으로 갈때마다 bn.activation.conv1블록으로 channel( = parameter) 줄이고
# 이때 first layer->second layer,first layer->third layer,first layer->four layer의 connection은 averagepooling을 통해 data shape 사이즈 조절

# Resnex
# resnex capacity 값이 클수록 효과 up -> cardinality = 16으로 병렬해보자


# dropout넣어 충분한 epoch를 들리게되면 더욱 효과가 좋아보일것으로 보임

x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
shortcut = 0
se = 0
Cf = 0.5
cardinality = 16

# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2
# 443
# --------------------------------------
# shortcut_1_4 = x
# shortcut_1_3 = x
# shortcut_1_2 = x
# shortcut_1_1 = x

shortcut_dense = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

# ------------------ first layer-1
# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

first_layer = []
for i in range(cardinality):

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)   # residual connection이 아니니까 activation 활성화시켜보자
    first_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer)
x = layers.Conv1D(64,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)


# 
se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(64 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(64, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut]) # x.shape = (100,256)


# ----------------- first layer-2

shortcut = x

# x = Concatenate()([x,shortcut_dense])

# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)
# x= layers.LeakyReLU(alpha = 0.01)(x)

# x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
# x = layers.BatchNormalization()(x)

first_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(4,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    first_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(first_layer2)
x = layers.Conv1D(64,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)


# x = Concatenate()([x,shortcut])


se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(64, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,100])(se)

x = Multiply()([x,se])


x = layers.Add()([x,shortcut])

# -----------------------------transition layer


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x = layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(64 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# ----------------- second layer-1

# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


second_layer = []
for i in range(cardinality):

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer)
x = layers.Conv1D(128,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(128 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(128, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])


# ----------------- second layer-2

shortcut = x

# x = Concatenate()([x,shortcut_dense])

second_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(8,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    second_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(second_layer2)
x = layers.Conv1D(128,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(128, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se])
x = layers.Add()([x,shortcut])

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(128 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# ----------------- third layer-1


# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

third_layer = []
for i in range(cardinality):

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer)
x = layers.Conv1D(256,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(256 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(256, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

# ----------------- third layer-2

shortcut = x

# x = Concatenate()([x,shortcut_dense])


third_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(16,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    third_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(third_layer2)
x = layers.Conv1D(256,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(256, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(64 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# --------------------------------------
# ----------------- four layer-1


# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


four_layer = []
for i in range(cardinality):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer)
x = layers.Conv1D(512,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(512 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(512, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

# ----------------- four layer-2

shortcut = x

# x = Concatenate()([x,shortcut_dense])

four_layer2 = []
for i in range(cardinality):

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)

    i = layers.Conv1D(32,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(i)
    i = layers.BatchNormalization()(i)
    i = layers.LeakyReLU(alpha = 0.01)(i)
    four_layer2.append(i)
# 기존 conv, pooling에서 kernel_size를 3x2로함
# 따라서 여기서도 kernel_size를 유지하기 위해 3x2로 함

x = Concatenate()(four_layer2)
x = layers.Conv1D(512,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(512, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])

x = Concatenate()([x,shortcut_dense])

# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)

model_se_res_dense_inceptionnet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_se_res_dense_inceptionnet.summary())


# In[ ]:


plot_model(model_se_res_dense_inceptionnet,show_shapes = True)


# In[ ]:


model_se_res_dense_inceptionnet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[ ]:


#  콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_se_res_dense_inceptionnet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[ ]:


model_se_res_dense_inceptionnet=model_se_res_dense_inceptionnet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [ early_stopping,model_checkpoint, reduce_lr],
                                                                   shuffle = True)


# In[ ]:


for key in model_se_res_dense_inceptionnet.history.keys():
    print(key)
  


# In[ ]:



plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(model_se_res_dense_inceptionnet.history['loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet.history['val_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(model_se_res_dense_inceptionnet.history['total_center3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet.history['val_total_center3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()
plt.ylim(0,1)

plt.subplot(234)
plt.plot(model_se_res_dense_inceptionnet.history['total_width3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet.history['val_total_width3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(model_se_res_dense_inceptionnet.history['total_amp3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet.history['val_total_amp3_loss'], 'r:', label = 'light -SE-Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[ ]:


plt.plot(model_se_res_dense_inceptionnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(model_se_res_dense_inceptionnet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.03)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[ ]:



from tensorflow.keras.models import load_model

best_model_se_res_dense_inceptionnet = load_model('best_model_se_res_dense_inceptionnet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_res_dense_inceptionnet.summary()


# In[ ]:



prediction_se_res_dense_inceptionnet = best_model_se_res_dense_inceptionnet.predict(test_peak)
print(len(prediction_se_res_dense_inceptionnet))
print(np.array(prediction_se_res_dense_inceptionnet).shape)


# In[ ]:


x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_se_res_dense_inceptionnet[0][i][0],prediction_se_res_dense_inceptionnet[1][i][0],prediction_se_res_dense_inceptionnet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[ ]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_res_dense_inceptionnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_res_dense_inceptionnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_res_dense_inceptionnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_se_res_dense_inceptionnet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[ ]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)


# In[ ]:





# In[ ]:





# In[ ]:


# four layer 의 CF 의 256 이야 64가 아니라
# 수정해


# In[ ]:





# In[ ]:


############################### new densenet2

# densenet 에 compossition factor 사용
# transition layer 사용
# first, second, third, four layer만 연결


# In[383]:


# new densenet
x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
shortcut = 0
se = 0
Cf = 0.5

# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2
# 443
# --------------------------------------
# shortcut_1_4 = x
# shortcut_1_3 = x
# shortcut_1_2 = x
# shortcut_1_1 = x

shortcut_dense = x
shortcut = x
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


# x = layers.LeakyReLU(alpha = 0.01)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)

# x = layers.LeakyReLU(alpha = 0.01)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)

# x = Concatenate()([x,x_1])

# 
se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(64 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(64, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut]) # x.shape = (100,256)

shortcut = x
# x = layers.LeakyReLU(alpha = 0.01)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)

# x = layers.LeakyReLU(alpha = 0.01)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)

# x = Concatenate()([x,shortcut])


se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(64, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,100])(se)

x = Multiply()([x,se])


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)

# -----------------------------transition layer


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x = layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(64 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# shortcut_2_4 = x
# shortcut_2_3 = x
# shortcut_2_2 = x
# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(128 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(128, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x
x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   #identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   #identity shortcut
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(128, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se])
x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(128 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)


# --------------------------------------
# shortcut_3_4 = x
# shortcut_3_3 = x

# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(256 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(256, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x
x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(256, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)

# transition layer---------------------------------


x = Concatenate()([x,shortcut_dense])
shortcut_dense = x
shortcut_dense = AveragePooling1D(3, padding = 'same', strides=2)(shortcut_dense)

# x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(256 * Cf,1, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = AveragePooling1D(3, padding = 'same', strides=2)(x)

# --------------------------------------
# shortcut_4_4 = x

# shortcut_dense = x

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(512 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(512, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])

shortcut = x
x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(512, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)

x = Concatenate()([x,shortcut_dense])

# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

# dropout_center = layers.Dropout(0.3)(x)
total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)


model_se_res_densenet2 = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_se_res_densenet2.summary())


# In[384]:


plot_model(model_se_res_densenet2,show_shapes = True)


# In[385]:


model_se_res_densenet2.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[386]:


#  콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_se_res_densenet2.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[387]:


model_se_res_densenet2=model_se_res_densenet2.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [ early_stopping, model_checkpoint, reduce_lr],
                                                 shuffle = True)


# In[388]:


for key in model_se_res_densenet2.history.keys():
    print(key)
  


# In[389]:



plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(model_se_res_densenet2.history['loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_densenet2.history['val_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(model_se_res_densenet2.history['total_center3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_densenet2.history['val_total_center3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()
plt.ylim(0,1)

plt.subplot(234)
plt.plot(model_se_res_densenet2.history['total_width3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_densenet2.history['val_total_width3_loss'], 'r:', label = 'light-SE-Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(model_se_res_densenet2.history['total_amp3_loss'], 'b-', label = 'light-SE-Resnet - training')
plt.plot(model_se_res_densenet2.history['val_total_amp3_loss'], 'r:', label = 'light -SE-Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[412]:


plt.plot(model_se_res_densenet2.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(model_se_res_densenet2.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.05)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[391]:



from tensorflow.keras.models import load_model

best_model_se_res_densenet2 = load_model('best_model_se_res_densenet2.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_res_densenet2.summary()


# In[392]:



prediction_se_res_densenet2 = best_model_se_res_densenet2.predict(test_peak)
print(len(prediction_se_res_densenet2))
print(np.array(prediction_se_res_densenet2).shape)


# In[393]:


x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_se_res_densenet2[0][i][0],prediction_se_res_densenet2[1][i][0],prediction_se_res_densenet2[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[394]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_res_densenet2[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_res_densenet2[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_res_densenet2[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_se_res_densenet2[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[395]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)

# 0.08672534273234021
# 0.03001733967329003
# 0.01918814255669618
# 0.00591970166100396


# In[ ]:


# 0.07020998659476517
# 0.02610263180185435
# 0.01743550979260114

# 0.07286502947507743
# 0.02647593086489068
# 0.017455254088738784

# 0.08222980937169654
# 0.02628184632921319
# 0.01781800222558519

# 0.07668137757324861
# 0.026323926194098107
# 0.016979520338370094


# In[ ]:


# ############################# Densenet 1

# densenet 에 compossition factor 사용
# transition layer 사용
# first, second, third, four layer만 연결
# first-1, first-2 연결

# 0.07847984827716617
# 0.02884114792175004
# 0.01755429039215598


# In[ ]:





# In[ ]:


############################### SE-resnet############################


# In[367]:


x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
shortcut = 0
se = 0

# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2
# 443
# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(64 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(64, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])
# x= layers.LeakyReLU(alpha = 0.01)(x)

shortcut = x
x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(64, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x,se])


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation


# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(128,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(128 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(128, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])
# x= layers.LeakyReLU(alpha = 0.01)(x)

shortcut = x
x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   #identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(128, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation


# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(256,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(256 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(256, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])
# x= layers.LeakyReLU(alpha = 0.01)(x)

shortcut = x
x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(256, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation

# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(512,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x) # global pooling
se = Dense(512 // r, kernel_initializer = 'he_normal')(se) # FC
se = layers.LeakyReLU(alpha = 0.01)(se) # ReLU
se = Dense(512, kernel_initializer = 'he_normal')(se) # FC
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se]) # Scale

x = layers.Add()([x,shortcut])
# x= layers.LeakyReLU(alpha = 0.01)(x)

shortcut = x
x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer = 'he_normal')(se)
se = layers.LeakyReLU(alpha = 0.01)(se)
se = Dense(512, kernel_initializer = 'he_normal')(se)
se = Activation('sigmoid')(se) # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x,se])

x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)


# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)


model_se_resnet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_se_resnet.summary())


# In[368]:


# Total params: 4,457,059
# Trainable params: 4,445,707
# Non-trainable params: 11,352
plot_model(model_se_resnet,show_shapes = True)


# In[369]:


model_se_resnet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[370]:


#  콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_se_resnet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[371]:


models_se_resnet=model_se_resnet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr],
                                    shuffle = True)


# In[372]:


for key in models_se_resnet.history.keys():
    print(key)
  


# In[373]:



plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(models_se_resnet.history['loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_se_resnet.history['val_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(models_se_resnet.history['total_center3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_se_resnet.history['val_total_center3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()
plt.ylim(0,1)

plt.subplot(234)
plt.plot(models_se_resnet.history['total_width3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_se_resnet.history['val_total_width3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(models_se_resnet.history['total_amp3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_se_resnet.history['val_total_amp3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[413]:


plt.plot(models_se_resnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_se_resnet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.05)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[375]:


# se-resnet

from tensorflow.keras.models import load_model

best_model_se_resnet = load_model('best_model_se_resnet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_resnet.summary()


# In[376]:


# se-resnet

prediction_se_resnet = best_model_se_resnet.predict(test_peak)
print(len(prediction_se_resnet))
print(np.array(prediction_se_resnet).shape)


# In[377]:


# se-resnet

x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_se_resnet[0][i][0],prediction_se_resnet[1][i][0],prediction_se_resnet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[378]:


# se-resnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_resnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_resnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_resnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_se_resnet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[379]:


# se-resnet

print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)
#10 by pationce = 0.5
# earlystop = 20


# In[ ]:


# 0.09928721694300995
# 0.030146564541407263
# 0.02074058037780464
#--------------------------------------------
# identity, projection 모두 activaiton no
# 0.08660375013222837
# 0.030121204554030903
# 0.0222768108861536


#--------------------------------------------
# projection,identity 모두 activation yes
# 0.0823119439125658
# 0.032458765038996334
# 0.02129520502020561


#--------------------------------------------
# projection 이전에만 activation - yes
# 0.08010383955473704
# 0.029172695294170114
# 0.021733901187878185

# 0.07907564682709822
# 0.02878694123025449
# 0.018642347253436133

# 0.08137700386793714
# 0.027786662827279224
# 0.01923721798146371


# In[ ]:


############################# resnet ###########################3


# In[ ]:


# import tensorflow as tf
# tf.device('/gpu:0')


# In[396]:


x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))
r = 16
# /gpu:0
# resnet 1차
x = layers.Conv1D(32,4,strides=2 ,padding = 'same',kernel_initializer = 'he_normal')(input_data)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,4,strides=1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.Conv1D(32,3,strides=1,  padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)
x = layers.MaxPooling1D(3, strides = 2)(x)  # 나누기 2

# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut) # projection shortcut,full pre -activation
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
shortcut = layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(64,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)


x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])

shortcut = x #identity shortcut
x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(64,3, strides = 1, padding = 'same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation


# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut,full pre -activation
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)  # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(128,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(128,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])

shortcut = x #identity shortcut
x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   #identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(128,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) #- full pre -activation


# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut,full pre -activation
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)   # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(256,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(256,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])

shortcut = x #identity shortcut
x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(256,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x) 

# --------------------------------------

shortcut = x
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut,full pre -activation
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 2, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)
shortcut = layers.BatchNormalization()(shortcut)    # projection shortcut
shortcut= layers.LeakyReLU(alpha = 0.01)(shortcut)
shortcut = layers.Conv1D(512,1,strides = 1, padding = 'valid',kernel_initializer = 'he_normal')(shortcut)

x = layers.Conv1D(512,3, strides = 2, padding='same',kernel_initializer = 'he_normal')(x)   # 나누기 2
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])

shortcut = x #identity shortcut
x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)   # identity shortcut
x = layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha = 0.01)(x)

x = layers.Conv1D(512,3, strides = 1, padding='same',kernel_initializer = 'he_normal')(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x,shortcut])
x= layers.LeakyReLU(alpha = 0.01)(x)


# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)

model_resnet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_resnet.summary())


# In[ ]:





# In[397]:


# from keras.utils import plot_model
# from tensorflow.keras.utils import plot_model

plot_model(model_resnet,show_shapes = True)


# In[398]:


model_resnet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[399]:


# model = keras.utils.multi_gpu_model(model, gpus=2)


# In[400]:


# 콜백설정
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_resnet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[ ]:


val_peak.shape


# In[ ]:


# x = np.linspace(-3,3,100)
# # t = []
# # for i in range(x):
# y = 1/1+np.exp(-x)
# z = 1-y
# max(z*y)


# plt.plot(x,y)


# In[ ]:


# x
# y = 1 / (1+ np.exp(-x))
# z = 1-y

# plt.plot(x,z*y)
# plt.ylim(-0.1,1)


# In[ ]:


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


# In[ ]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[401]:


models_resnet=model_resnet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp),np.array(val_peak_number)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[402]:


#resnet

for key in models_resnet.history.keys():
    print(key)


# In[403]:


#resnet
plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(models_resnet.history['loss'], 'b-', label = 'Resnet - training')
plt.plot(models_resnet.history['val_loss'], 'r:', label = 'Resnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()


# plt.subplot(232)
# plt.plot(models.history['Dense3_peak_number_loss'], 'b-', label = 'training')
# plt.plot(models.history['val_Dense3_peak_number_loss'], 'r:', label = 'validation')
# plt.grid(True)
# plt.title("Number of Peak Loss", size = 32)
# plt.legend()

plt.subplot(232)
plt.plot(models_resnet.history['total_center3_loss'], 'b-', label = 'Resnet - training')
plt.plot(models_resnet.history['val_total_center3_loss'], 'r:', label = 'Resnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()

plt.subplot(234)
plt.plot(models_resnet.history['total_width3_loss'], 'b-', label = 'Resnet - training')
plt.plot(models_resnet.history['val_total_width3_loss'], 'r:', label = 'Resnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(models_resnet.history['total_amp3_loss'], 'b-', label = 'Resnet - training')
plt.plot(models_resnet.history['val_total_amp3_loss'], 'r:', label = 'Resnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[414]:


plt.plot(models_resnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_resnet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.05)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[405]:


#resnet

from tensorflow.keras.models import load_model

best_model_resnet = load_model('best_model_resnet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_resnet.summary()


# In[406]:


#resnet

prediction_resnet = best_model_resnet.predict(test_peak)
print(len(prediction_resnet))
print(np.array(prediction_resnet).shape)


# In[407]:


#resnet

x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_resnet[0][i][0],prediction_resnet[1][i][0],prediction_resnet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[408]:


#resnet


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_resnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_resnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_resnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_resnet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[409]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)
# 1.1


# In[ ]:


########################### vggnet ########################


# In[415]:


# vggnet

x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))

x = layers.Conv1D(32,4,strides=2, activation = 'relu',padding = 'same')(input_data)
x = layers.Conv1D(32,4,strides=1, activation = 'relu', padding='same')(x)
x = layers.Conv1D(32,3,strides=1, activation = 'relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides = 2)(x)

x = layers.Conv1D(64,3,strides=1, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(64,3,strides=1, activation = 'relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides = 2)(x)

x = layers.Conv1D(128,3,strides=1, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(128,3,strides=1, activation = 'relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides = 2)(x)

x = layers.Conv1D(256,3,strides=1, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(256,3,strides=1, activation = 'relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2,strides = 2)(x)

x = layers.Conv1D(512,3,strides=1, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(512,3,strides=1, activation = 'relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2,strides = 2)(x)

x = layers.GlobalMaxPooling1D()(x)



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)

model_vggnet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_vggnet.summary())


# In[416]:


# vggnet
plot_model(model_vggnet,show_shapes = True)


# In[417]:


# vggnet
# best_model3.h5

model_vggnet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[418]:


# vggnet

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_vggnet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[419]:


# vggnet

models_vggnet=model_vggnet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp),np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[420]:


# vggnet

for key in models_vggnet.history.keys():
    print(key)


# In[421]:


# vggnet
plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(models_vggnet.history['loss'], 'b-', label = 'VGGnet - training')
plt.plot(models_vggnet.history['val_loss'], 'r:', label = 'VGGnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(models_vggnet.history['total_center3_loss'], 'b-', label = 'VGGnet - training')
plt.plot(models_vggnet.history['val_total_center3_loss'], 'r:', label = 'VGGnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()

plt.subplot(234)
plt.plot(models_vggnet.history['total_width3_loss'], 'b-', label = 'VGGnet - training')
plt.plot(models_vggnet.history['val_total_width3_loss'], 'r:', label = 'VGGnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(models_vggnet.history['total_amp3_loss'], 'b-', label = 'VGGnet - training')
plt.plot(models_vggnet.history['val_total_amp3_loss'], 'r:', label = 'VGGnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[422]:


plt.plot(models_vggnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_vggnet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.03)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[423]:


# vggnet

from tensorflow.keras.models import load_model

best_model_vggnet = load_model('best_vggnet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_vggnet.summary()


# In[424]:


# vggnet

prediction_vggnet = best_model_vggnet.predict(test_peak)
print(len(prediction_vggnet))
print(np.array(prediction_vggnet).shape)


# In[425]:


# vggnet

x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_vggnet[0][i][0],prediction_vggnet[1][i][0],prediction_vggnet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[427]:


# vggnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_vggnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_vggnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_vggnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_vggnet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[428]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)


# In[ ]:


####################### Alex+ZFnet ###################


# In[429]:


# alexnet+zfnet
x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))


x = layers.Conv1D(96,20,strides=2,activation = 'relu',padding = 'same')(input_data)
x = layers.MaxPooling1D(3,strides = 2,padding = 'same')(x)
x = layers.Conv1D(256,9,strides = 2, activation = 'relu',padding = 'same')(x)
x = layers.MaxPooling1D(3,strides = 2,padding = 'same')(x)
x = layers.Conv1D(384,4, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(384,4, activation = 'relu',padding = 'same')(x)
x = layers.Conv1D(256,3, activation = 'relu',padding = 'same')(x)
x = layers.MaxPooling1D(3,strides = 2,padding = 'same')(x)
x = layers.GlobalMaxPooling1D()(x)



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)

total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)


model_alex_zfnet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_alex_zfnet.summary())


# In[430]:


# alexnet+zfnet
plot_model(model_alex_zfnet,show_shapes = True)


# In[431]:


# alexnet+zfnet

model_alex_zfnet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[432]:


# alexnet+zfnet

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_alex_zfnet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[433]:


# alexnet+zfnet

models_alex_zfnet=model_alex_zfnet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp), np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[434]:


# alexnet+zfnet

for key in models_alex_zfnet.history.keys():
    print(key)


# In[435]:


# alexnet+zfnet
plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(models_alex_zfnet.history['loss'], 'b-', label = 'Alex_ZFnet - training')
plt.plot(models_alex_zfnet.history['val_loss'], 'r:', label = 'Alex_ZFnet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(models_alex_zfnet.history['total_center3_loss'], 'b-', label = 'Alex_ZFnet - training')
plt.plot(models_alex_zfnet.history['val_total_center3_loss'], 'r:', label = 'Alex_ZFnet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()

plt.subplot(234)
plt.plot(models_alex_zfnet.history['total_width3_loss'], 'b-', label = 'Alex_ZFnet - training')
plt.plot(models_alex_zfnet.history['val_total_width3_loss'], 'r:', label = 'Alex_ZFnet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(models_alex_zfnet.history['total_amp3_loss'], 'b-', label = 'Alex_ZFnet - training')
plt.plot(models_alex_zfnet.history['val_total_amp3_loss'], 'r:', label = 'Alex_ZFnet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[436]:


plt.plot(models_alex_zfnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_alex_zfnet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.03)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[437]:


# alexnet+zfnet

from tensorflow.keras.models import load_model

best_model_alex_zfnet = load_model('best_model_alex_zfnet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_alex_zfnet.summary()


# In[438]:


# alexnet+zfnet

prediction_alex_zfnet = best_model_alex_zfnet.predict(test_peak)
print(len(prediction_alex_zfnet))
print(np.array(prediction_alex_zfnet).shape)


# In[439]:


# alexnet+zfnet

x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_alex_zfnet[0][i][0],prediction_alex_zfnet[1][i][0],prediction_alex_zfnet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[440]:


# alexnet+zfnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_alex_zfnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_alex_zfnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_alex_zfnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_alex_zfnet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[441]:


# alexnet+zfnet

print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)


# In[ ]:


########################### lenet#################################
###########################before peak_fitting2 project model#######


# In[353]:


x = np.linspace(0,15,401)

input_data = Input(shape = (len(x), 1))

x = layers.Conv1D(32,100,strides=3,activation = 'relu')(input_data)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(64,10,strides = 2,activation = 'relu')(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(128,4,activation = 'relu')(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(256,2,activation = 'relu')(x)
x = layers.MaxPooling1D(2)(x)

x = layers.GlobalMaxPooling1D()(x)



total_center1 = Dense(100,name = 'total_center1',kernel_initializer = 'he_normal')(x)
center_Batchnormalization  = BatchNormalization()(total_center1)
total_center1_act= layers.LeakyReLU(alpha = 0.01)(center_Batchnormalization)
total_center3 = Dense(1, activation = 'linear',name = 'total_center3',kernel_initializer = 'he_normal')(total_center1_act)

total_width1 = Dense(100, name = 'total_width1',kernel_initializer = 'he_normal')(x)
width_Batchnormalization  = BatchNormalization()(total_width1)
total_width1_act= layers.LeakyReLU(alpha = 0.01)(width_Batchnormalization)
total_width3= Dense(1, activation = 'linear',name = 'total_width3',kernel_initializer = 'he_normal')(total_width1_act)

total_amp1 = Dense(100,name = 'total_amp1',kernel_initializer = 'he_normal')(x)
amp_Batchnormalization  = BatchNormalization()(total_amp1)
total_amp1_act= layers.LeakyReLU(alpha = 0.01)(amp_Batchnormalization)
total_amp3 = Dense(1, activation = 'linear',name = 'total_amp3',kernel_initializer = 'he_normal')(total_amp1_act)


total_peak_number1 = Dense(100,name = 'total_peak_number1',kernel_initializer = 'he_normal')(x)
peak_number_Batchnormalization  = BatchNormalization()(total_peak_number1)
total_peak_number1_act= layers.LeakyReLU(alpha = 0.01)(peak_number_Batchnormalization)
total_peak_number3 = Dense(1, activation = 'linear',name = 'total_peak_number3',kernel_initializer = 'he_normal')(total_peak_number1_act)

model_lenet = Model(inputs = input_data,
              outputs = [total_center3,total_width3,total_amp3,total_peak_number3])
print(model_lenet.summary())


# In[354]:


plot_model(model_lenet,show_shapes = True)


# In[355]:



model_lenet.compile(optimizer='adam',
              loss = {'total_center3' : 'mse',
                      'total_width3' : 'mse',
                      'total_amp3' : 'mse',
                      'total_peak_number3' : 'mse'},
              loss_weights={'total_center3' : 8,
                           'total_width3' : 53,
                           'total_amp3' : 91,
                            'total_peak_number3' :3.3 },
              metrics = ['mae'])


# In[356]:


# lenet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model_checkpoint = ModelCheckpoint('best_model_lenet.h5', save_best_only = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1 )
# 4 7


# In[357]:


# lenet
models_lenet=model_lenet.fit(train_peak, [np.array(train_center) , np.array(train_width), np.array(train_amp), np.array(train_peak_number)],
          epochs = 50,
          batch_size = 512,
          validation_data = (val_peak, [np.array(val_center), np.array(val_width), np.array(val_amp), np.array(val_peak_number)]),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])


# In[358]:


for key in models_lenet.history.keys():
    print(key)


# In[359]:


plt.figure(figsize = (25, 15))

plt.subplot(231)
plt.plot(models_lenet.history['loss'], 'b-', label = 'Lenet - training')
plt.plot(models_lenet.history['val_loss'], 'r:', label = 'Lenet - validation')
plt.grid(True)
plt.title("Total Loss", size = 32)
plt.legend()



plt.subplot(232)
plt.plot(models_lenet.history['total_center3_loss'], 'b-', label = 'Lenet - training')
plt.plot(models_lenet.history['val_total_center3_loss'], 'r:', label = 'Lenet - validation')
plt.grid(True)
plt.title("center Loss", size = 32)
plt.legend()

plt.subplot(234)
plt.plot(models_lenet.history['total_width3_loss'], 'b-', label = 'Lenet - training')
plt.plot(models_lenet.history['val_total_width3_loss'], 'r:', label = 'Lenet - validation')
plt.grid(True)
plt.title("width Loss", size = 32)
plt.legend()

plt.subplot(235)
plt.plot(models_lenet.history['total_amp3_loss'], 'b-', label = 'Lenet - training')
plt.plot(models_lenet.history['val_total_amp3_loss'], 'r:', label = 'Lenet - validation')
plt.grid(True)
plt.title("amp Loss", size = 32)
plt.legend()


# In[360]:


plt.plot(models_lenet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training')
plt.plot(models_lenet.history['val_total_peak_number3_loss'], 'r:', label = 'SE-Resnet - validation')
plt.grid(True)
plt.ylim(0,0.03)
plt.title("peak number Loss", size = 32)
plt.legend()


# In[361]:


from tensorflow.keras.models import load_model

best_model_lenet = load_model('best_model_lenet.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_lenet.summary()


# In[362]:


prediction_lenet = best_model_lenet.predict(test_peak)
print(len(prediction_lenet))
print(np.array(prediction_lenet).shape)


# In[363]:


x = np.linspace(0,15,401)

for i in range(50,100):
    plt.figure(figsize = (10,5))
    plt.plot(x,test_peak[i], c = 'black')
#     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(x,y(prediction_lenet[0][i][0],prediction_lenet[1][i][0],prediction_lenet[2][i][0],x),label = 'predict',c = 'red')
    
    plt.plot(x,y(test_center[i],test_width[i],test_amp[i],x),c = 'blue', label = 'real')
#     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[365]:



loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_lenet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_lenet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_lenet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_lenet[3][i][0]))
    
loss_center = loss_center/len(test_center)
loss_width = loss_width/len(test_center)
loss_amp = loss_amp/len(test_center)
loss_peak_number = loss_peak_number/len(test_center)


# In[366]:


print(loss_center)
print(loss_width)
print(loss_amp)
print(loss_peak_number)


# In[ ]:


# 0.17258414328986452
# 0.051134848152052575
# 0.034923534239341254


# In[455]:


plt.figure(figsize = (25, 15))

# total_train_loss
plt.subplot(231)
plt.plot(model_se_res_densenet2.history['loss'], 'b-', label = 'SE-Res-Densenet - training')
plt.plot(models_se_resnet.history['loss'], 'b-', label = 'SE-Resnet - training', c = 'm')
plt.plot(models_resnet.history['loss'], 'b-', label = 'Resnet - training', c = 'red')
plt.plot(models_vggnet.history['loss'], 'b-', label = 'VGGnet - training', c = 'firebrick')
plt.plot(models_alex_zfnet.history['loss'], 'b-', label = 'Alex-ZFnet - training', c = 'sandybrown')
plt.plot(models_lenet.history['loss'], 'b-', label = 'Lenet - training', c = 'gold')
plt.grid(True)
plt.title("Total Train Loss", size = 32)
plt.legend()


# total_val_loss
plt.subplot(232)
plt.plot(model_se_res_densenet2.history['val_loss'], 'b-', label = 'SE-Res-Densenet - validation')
plt.plot(models_se_resnet.history['val_loss'], 'b-', label = 'SE-Resnet - validation', c = 'm')
plt.plot(models_resnet.history['val_loss'], 'b-', label = 'Resnet - validation', c = 'red')
plt.plot(models_vggnet.history['val_loss'], 'b-', label = 'VGGnet - validation', c = 'firebrick')
plt.plot(models_alex_zfnet.history['val_loss'], 'b-', label = 'Alex-ZFnet - validation', c = 'sandybrown')
plt.plot(models_lenet.history['val_loss'], 'b-', label = 'Lenet - validation', c = 'gold')
plt.grid(True)
plt.title("Total Validation Loss", size = 32)
plt.legend()
plt.ylim(0,22.5)

# center_train_loss
plt.subplot(234)
plt.plot(model_se_res_densenet2.history['total_center3_loss'], 'b-', label = 'SE-Res-Densenet - training')
plt.plot(models_se_resnet.history['total_center3_loss'], 'b-', label = 'SE-Resnet - training',c = 'm')
plt.plot(models_resnet.history['total_center3_loss'], 'b-', label = 'Resnet - training', c = 'red')
plt.plot(models_vggnet.history['total_center3_loss'], 'b-', label = 'VGGnet - training', c = 'firebrick')
plt.plot(models_alex_zfnet.history['total_center3_loss'], 'b-', label = 'Alex-ZFnet - training', c = 'sandybrown')
plt.plot(models_lenet.history['total_center3_loss'], 'b-', label = 'Lenet - training', c = 'gold')
plt.grid(True)
plt.title("Train Center Loss", size = 32)
plt.legend()

# center_val_loss
plt.subplot(235)
plt.plot(model_se_res_densenet2.history['val_total_center3_loss'], 'b-', label = 'SE-Res-Densenet - validation')
plt.plot(models_se_resnet.history['val_total_center3_loss'], 'b-', label = 'SE-Resnet - validation', c = 'm')
plt.plot(models_resnet.history['val_total_center3_loss'], 'b-', label = 'Resnet - validation', c = 'red')
plt.plot(models_vggnet.history['val_total_center3_loss'], 'b-', label = 'VGGnet - validation', c = 'firebrick')
plt.plot(models_alex_zfnet.history['val_total_center3_loss'], 'b-', label = 'Alex-ZFnet - validation', c = 'sandybrown')
plt.plot(models_lenet.history['val_total_center3_loss'], 'b-', label = 'Lenet - validation', c = 'gold')
plt.grid(True)
plt.title("Validation Center Loss", size = 32)
plt.legend()


# In[480]:


plt.figure(figsize = (25, 25))

plt.subplot(211)
plt.plot(model_se_res_densenet2.history['total_peak_number3_loss'], 'b-', label = 'SE-Res-Densenet - training')
plt.plot(models_se_resnet.history['total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training',c = 'm')
plt.plot(models_resnet.history['total_peak_number3_loss'], 'b-', label = 'Resnet - training', c = 'red')
plt.plot(models_vggnet.history['total_peak_number3_loss'], 'b-', label = 'VGGnet - training', c = 'firebrick')
plt.plot(models_alex_zfnet.history['total_peak_number3_loss'], 'b-', label = 'Alex-ZFnet - training', c = 'sandybrown')
plt.plot(models_lenet.history['total_peak_number3_loss'], 'b-', label = 'Lenet - training', c = 'gold')
plt.grid(True)
plt.title("Train Center Loss", size = 32)
plt.legend()
plt.ylim(0,0.1)

plt.subplot(212)
plt.plot(model_se_res_densenet2.history['val_total_peak_number3_loss'], 'b-', label = 'SE-Res-Densenet - training')
plt.plot(models_se_resnet.history['val_total_peak_number3_loss'], 'b-', label = 'SE-Resnet - training',c = 'm')
plt.plot(models_resnet.history['val_total_peak_number3_loss'], 'b-', label = 'Resnet - training', c = 'red')
plt.plot(models_vggnet.history['val_total_peak_number3_loss'], 'b-', label = 'VGGnet - training', c = 'firebrick')
plt.plot(models_alex_zfnet.history['val_total_peak_number3_loss'], 'b-', label = 'Alex-ZFnet - training', c = 'sandybrown')
plt.plot(models_lenet.history['val_total_peak_number3_loss'], 'b-', label = 'Lenet - training', c = 'gold')
plt.grid(True)
plt.title("Train Center Loss", size = 32)
plt.legend()
plt.ylim(0,0.01)


# In[ ]:


# resnet
# 0.10816401457979095
# 0.030619180051451405
# 0.020286300825483432

# se-resnet
# 0.08335333617545047
# 0.030164464011837484
# 0.020591724598516287

# 0.08119587027995552
# 0.02853893844762173
# 0.019406292207859128


# In[885]:


# 0.1980241036026108
# 0.0434090221804756
# 0.03321981394717362

# lenet
# noise = 0.01 ,batch_size = 200, epoch = 100, lr = 0.5
#기준 area

# 0.2876684894293567
# 0.033030751748916974
# 0.023151721246088306

# zfnet
# noise = 0.01 ,batch_size = 200, epoch = 100, lr = 0.5
#기준 area

# 0.11264473291117566
# 0.021803617173060594
# 0.01740119580528536

# ---------------------------------------- p3ht그래프 시각화 수정
# zfnet
# noise = 0.01 ,batch_size = 500, epoch = 50, lr = 0.1
# 0.11684046318704204
# 0.02365474199326388
# 0.017884804965016435
#기준 area

# zfnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
#기준 area

# 0.14462273796028452
# 0.031952290796258394
# 0.02227899902482363

# zfnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
# 기준 center

# 0.0403674395524352
# 0.02615626201945091
# 0.02677103315473494
#파라미더 1,580,000

#resnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
#기준 area

# 0.11857149868883526
# 0.03263180739217448
# 0.022952831307554454

# -----------------------------------------new data
# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1 feature 32
#기준 area
# 0.11484013720310739
# 0.03064794644125652
# 0.02168992991763753

# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1 feature 64
#기준 area
# 0.17572729285820568
# 0.029032754705661946
# 0.020763448977790314

# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 10, lr = 0.1 feature 64
#기준 area
# 0.17314537565227558
# 0.028766554175357072
# 0.021212849061068805

#resnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1
# ,projection shortcut 1
#기준 area

# 0.219934999018171
# 0.041640542136089184
# 0.028254922584841143

# resnet 18layer
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1

# 0.11579423334714921
# 0.03162808413659801
# 0.021646715531842685

# resnet 18layer
# leakyrelu(0.01), kernel_init = he_normal,384
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1
# not pre-activation

# 0.11236779294782767
# 0.030167100940907545
# 0.02108751209296305

# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# vgg(plain)net 
# 0.11908134603087908
# 0.029694876976764893
# 0.02220266332971771

# plainet + projection shortcut 1layer
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# 0.14454205265157546
# 0.035656453286206834
# 0.02518174399651547

# plainet+ projection shortcut 2layer
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# 실수의 maxpooling and strides 2
# 최종 output kernelsize 7

# 0.13743766259137466
# 0.034960850331634034
# 0.024586934629632767

# resnet original
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
#layers 19
# pre activation

# 0.10910127349335477
# 0.029502006677230425
# 0.02118154023550757


# In[996]:


x = np.linspace(0,15,401)
# x = np.linspace(0,7,401)
t = np.linspace(0,401,401)


# In[997]:


# only three peaks

bg = np.loadtxt("ITO_O1s_bg.txt")
exp = np.loadtxt("ITO_O1s_exp.txt")
fitting = np.loadtxt("ITO_O1s_fitting.txt")
peak1 = np.loadtxt("ITO_O1s_p1.txt")
peak2 = np.loadtxt("ITO_O1s_p2.txt")
peak3 = np.loadtxt("ITO_O1s_p3.txt")

# 테스트하고자 하는 실제 XPS 데이터의 parameter 범위가 너무 크므로 
# 네트워크 자체는 작은 규모의 파라미터 범위에서 학습시키고,
# 테스트할 경우, 범위를 줄인 xps 데이터를 불러와서 테스트해본다

plt.figure(figsize = (10,5))
# exp data에서 background를 제거하고, peak 높이를 1로 normalize한다.
plt.plot(exp[:, 0], (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")

#plt.plot(fitting[:, 0], fitting[:, 1] - bg[:, 1], label = "fitting", linewidth = 2)

# 마찬가지로 개별 peak도 크기를 줄인다.
plt.plot(peak1[:, 0], (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak1", linewidth = 2)
plt.plot(peak2[:, 0], (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak2", linewidth = 2)
plt.plot(peak3[:, 0], (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'peak3', linewidth = 2)

plt.grid(True)
plt.title("P3HT Fitting and experiment", size = 24)
plt.xlabel("Energy range", size = 24)
plt.ylabel("Intensity", size = 24)
plt.legend()
plt.show()


# In[998]:


((exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()))[-1:-402:-1].shape


# In[999]:


np.array((1,2,3))[-1:-4:-1]


# In[1000]:


test_result = ((exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max())).reshape((1, 401, 1))
print(test_result.shape)
# plt.plot( (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")
# plt.plot(exp[:, 0], (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")

plt.plot(test_result[0], c = 'black')
plt.title('total p3ht')


# In[1001]:


test_result.shape


# In[1002]:


x.dtype


# In[1003]:


# 첫번째 맨 오른쪽 peak 예측

predict = best_model.predict(test_result)
print(predict)


# In[1004]:


new_predict = []

for element in predict:
    new_predict.append(element.reshape((element.shape[1])))


# In[1005]:


new_predict


# In[1006]:


# plt.plot(test_result[0])
plt.plot(test_result[0], c = 'black', label = 'ito')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "real peak", linewidth = 2, color='purple')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),  linewidth = 2, color='purple')
plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),  linewidth = 2, color='purple')

plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),label = 'predict first peak', c = 'blue')
plt.title('first peak')
plt.legend()


# In[1007]:


test_result[0].reshape(401,).shape


# In[1008]:


plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),c = 'black')
plt.title('total peak without first peak')


# In[1009]:


(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)).shape


# In[1010]:



new_predict[0][0]
a = y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)

list(a).index(max(a))
len(a)-list(a).index(max(a))
range(list(a).index(max(a)) ,len(a),1)


# In[1011]:


np.random.rand() * noise_level - noise_level * 0.5


# In[1012]:


# 첫번째 보정
# center기준 오른쪽 값은 다 0처리
# 어짜피 예측값은 가장 오른쪽 center였으니까 더이상 없겠지

# 두번째 보정
# 음수값들은 다 0처리
# 보니까 음수값으로 굴곡이 휘어진걸 peak으로 구분함 따라서 0처리


test_result2 = test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)

a = y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)
print(list(a).index(max(a)))

# for j in range(list(a).index(max(a)), len(a), 1):
# #     test_result2[j] = np.random.rand() * noise_level - noise_level * 0.5
#     test_result2[j] = 0

for i in range(test_result2.shape[0]):
    
    if test_result2[i] < 0. :
#         test_result2[i] = np.random.rand() * noise_level - noise_level * 0.5
            test_result2[i] = 0
        
plt.plot(test_result2)        


# In[1013]:


test_result2 = test_result2.reshape(1,401,1)
test_result2.shape


# In[1014]:


# 2번째 맨 오른쪽 픽 예측

# 아 이값은 보정이 안되어 잇잖아
# test_result2 = (test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)).reshape(1,401,1)
# test_result2.shape

predict2 = best_model.predict(test_result2)
# print(predict2)

new_predict2 = []

for element in predict2:
    new_predict2.append(element.reshape((element.shape[1])))

# plt.plot(test_result2.reshape(401,),c = 'green')        
# plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),c = 'black')   
# plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', label = "peak2", linewidth = 2)
plt.plot(test_result[0], c = 'black', label = 'ito')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "real peak", linewidth = 2, color='purple')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), linewidth = 2, color='purple')
plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),  linewidth = 2, color='purple')


# plt.plot(test_result2.reshape(401,), c = 'dodgerblue')   
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),label = 'predict second peak', c = 'blue')
plt.title('second peak')
plt.legend()


# In[1015]:


plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),c = 'black')
plt.title('total peak without second  peak')


# In[1016]:


test_result3 = test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x)

a = y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x)
print(list(a).index(max(a)))

# for j in range(list(a).index(max(a)), len(a), 1):
#     test_result3[j] = 0


for i in range(test_result3.shape[0]):
    if test_result3[i] < 0. :
        test_result3[i] = 0
        
plt.plot(test_result3)    


# In[1017]:


test_result3 = test_result3.reshape(1,401,1)
test_result3.shape


# In[1018]:


# 3번째 맨 오른쪽픽 예측

# test_result3 = (test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x)).reshape(1,401,1)
# test_result3.shape

predict3 = best_model.predict(test_result3)
# print(predict3)

new_predict3 = []

for element in predict3:
    new_predict3.append(element.reshape((element.shape[1])))

plt.plot(test_result[0], c = 'black', label = 'ito')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "real peak", linewidth = 2, color='purple')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), linewidth = 2, color='purple')
plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), linewidth = 2, color='purple')

# plt.plot(test_result3.reshape(401,), c = 'dodgerblue')   
# plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),c = 'black')
# plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'peak3',color = 'purple', linewidth = 2)
plt.plot(t,y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'predict thrid peak', c = 'blue')
plt.title('third peak')
plt.legend()


# In[1019]:


plt.plot(test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x), c = 'black')
plt.title('total peak without thrid right peak')


# In[1020]:


test_result4 = test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x)
for i in range(test_result4.shape[0]):
    if test_result4[i] < 0. :
        test_result4[i] = 0
        
plt.plot(test_result4) 


# In[1021]:


test_result4 = test_result4.reshape(1,401,1)
test_result4.shape


# In[1022]:


# 없는 4peak을 예측하면 무슨값이 나올려나
# noise가 -값까지 가는구나. -값을 가지면 끝났다고 생각해도 되겠구나

test_result4 = (test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x)).reshape(1,401,1)
test_result4.shape

predict4 = best_model.predict(test_result4)
# print(predict4)

new_predict4 = []

for element in predict4:
    new_predict4.append(element.reshape((element.shape[1])))
# new_predict4
plt.plot(test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x), c = 'black')
plt.plot(t,y(new_predict4[0][0],new_predict4[1][0],new_predict4[2][0],x),label = 'predict peak1', c = 'blue')


# In[1023]:


plt.plot(test_result4[0].reshape(401,)- y(new_predict4[0][0],new_predict4[1][0],new_predict4[2][0],x))
plt.title('total peak without four  peak')


# In[ ]:





# In[1024]:


t = np.linspace(0,401,401)

plt.figure(figsize = (20,25))

plt.subplot(421)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.title('ito',size = 20)
plt.legend()

plt.subplot(423)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), c = 'blue', label = 'first predict')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2,label = 'first real')
plt.legend()
plt.title('compare first real, predict peak', size = 20)

plt.subplot(424)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), label = 'ito without first peak')
plt.title('ito without predict first peak', size = 20)
plt.legend()

plt.subplot(425)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict2[2][0],x),c = 'dodgerblue',label = 'ito withoutfirst peak')
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x), c = 'blue', label = 'second  predict')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2, label = 'second real')
plt.title('compare second real, predict peak', size = 20)
plt.legend()

plt.subplot(426)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict2[2][0],x), label = 'ito without first peak', c = 'dodgerblue')
plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),label = 'ito without predict first,second  peak', c = 'orange')
plt.title( 'ito without predict first,second peak', size = 20)
plt.legend()

plt.subplot(427)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
# plt.plot(test_result[0].reshape(401,)- y(new_predict[1][0],new_predict[2][0],new_predict2[3][0],x), label = 'p3ht without right first peak', c = 'dodgerblue')
plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict3[2][0],x),label = 'ito without predict first,second peak', c = 'orange')
plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'third real',color = 'purple', linewidth = 2)
plt.plot(t,y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'third predict', c = 'blue')
plt.title('compare third real, predict peak', size = 20)
plt.legend()

plt.subplot(428)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total ito' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict2[2][0],x), label = 'ito without first peak', c = 'dodgerblue')
plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),label = 'ito without predict first,second  peak', c = 'orange')
plt.plot(test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'ito without predict first,second,thrid  peak', c = 'royalblue')
plt.title( 'ito without predict first,second,third peak', size = 20)
plt.legend()



# In[1025]:



plt.figure(figsize = (10,5))

# x = np.linspace(0,401,401)
# t = np.linspace(0,7,401)

# x = np.linspace(0,15,401)
x = np.linspace(0,15,401)
t = np.linspace(0,401,401)


plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), c = 'blue')
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x), c = 'blue')
plt.plot(t,y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'predict', c = 'blue')


plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2)
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2)
plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'real',color = 'purple', linewidth = 2)
plt.title( 'ito with real, predict peaks', size = 20)

plt.plot(test_result[0], c = 'black', label = 'p3ht')
plt.legend()


# In[1026]:


plt.figure(figsize = (10,5))
plt.plot(test_result[0],label = 'total ito peak' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict2[2][0],x),label = 'ito without predict first peak')
plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict3[2][0],x),label = 'ito without predict first,second peak')
plt.plot(test_result3[0].reshape(401,)- y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'ito without predict first,second,thrid peak')

# x = np.linspace(0,401,400)
t = np.linspace(0,401,401)

# plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak1", linewidth = 2, color='black')
# plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak2", linewidth = 2, color='black')
# plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak3", linewidth = 2, color='black')
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), c = 'blue')
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x), c = 'blue')
plt.plot(t,y(new_predict3[0][0],new_predict3[1][0],new_predict3[2][0],x),label = 'predict', c = 'blue')

plt.legend()


# In[1027]:


bg = np.loadtxt("P3HT_S2p_bg.txt")
exp = np.loadtxt("P3HT_S2p_exp.txt")
fitting = np.loadtxt("P3HT_S2p_fitting.txt")
peak1 = np.loadtxt("P3HT_S2p_p1.txt")
peak2 = np.loadtxt("P3HT_S2p_p2.txt")


plt.figure(figsize = (10,5))
# exp data에서 background를 제거하고, peak 높이를 1로 normalize한다.
plt.plot(exp[:, 0], (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")

#plt.plot(fitting[:, 0], fitting[:, 1] - bg[:, 1], label = "fitting", linewidth = 2)

# 마찬가지로 개별 peak도 크기를 줄인다.
plt.plot(peak1[:, 0], (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak1", linewidth = 2)
plt.plot(peak2[:, 0], (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak2", linewidth = 2)
# plt.plot(peak3[:, 0], (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'peak3', linewidth = 2)

plt.grid(True)
plt.title("P3HT Fitting and experiment", size = 24)
plt.xlabel("Energy range", size = 24)
plt.ylabel("Intensity", size = 24)
plt.legend()
plt.show()


# In[1028]:


test_result = ((exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max())).reshape((1, 401, 1))
print(test_result.shape)
plt.plot(test_result[0], c = 'black')
plt.title('total p3ht')


# In[1029]:


# 첫번째 맨 오른쪽 peak 예측

predict = best_model.predict(test_result)
# print(predict)

new_predict = []

for element in predict:
    new_predict.append(element.reshape((element.shape[1])))

plt.plot(test_result[0], c = 'black')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = "peak1", linewidth = 2, color='purple')
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),label = 'predict first peak', c = 'blue')
plt.title('first peak')
plt.legend()


# In[1030]:


plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),c = 'black')
plt.title('total peak without first peak')


# In[1031]:


test_result2 = test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)
a = y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)
print(list(a).index(max(a)))

# for j in range(list(a).index(max(a)), len(a), 1):
# #     test_result2[j] = np.random.rand() * noise_level - noise_level * 0.5
#     test_result2[j] = 0


for i in range(test_result2.shape[0]):
    if test_result2[i] < 0. :
#         test_result2[i] = np.random.rand() * noise_level - noise_level * 0.5
        test_result2[i] = 0
        
plt.plot(test_result2)


# In[1032]:


test_result2 = test_result2.reshape(1,401,1)
test_result2.shape


# In[1033]:


# 2번째 맨 오른쪽 픽 예측

# test_result2 = (test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x)).reshape(1,401,1)
# test_result2.shape

predict2 = best_model.predict(test_result2)
# print(predict2)

new_predict2 = []

for element in predict2:
    new_predict2.append(element.reshape((element.shape[1])))
   
plt.plot(test_result[0], c = 'black')
plt.plot(test_result2.reshape(401,), c = 'dodgerblue')   
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', label = "peak2", linewidth = 2)
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),label = 'predict second peak', c = 'blue')
plt.title('second peak')
plt.legend()


# In[1034]:


plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),c = 'black')
plt.title('total peak without second peak')


# In[1035]:


t = np.linspace(0,401,401)

plt.figure(figsize = (20,25))

plt.subplot(421)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
plt.title('p3ht',size = 20)
plt.legend()

plt.subplot(423)
plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), c = 'blue', label = 'first predict')
plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2,label = 'first real')
plt.legend()
plt.title('compare first real, predict peak', size = 20)

plt.subplot(424)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), label = 'p3ht without first peak')
plt.title('p3ht without predict first peak', size = 20)
plt.legend()

plt.subplot(425)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),c = 'dodgerblue',label = 'p3ht without first peak')
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x), c = 'blue', label = 'second predict')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2, label = 'second real')
plt.title('compare second real, predict peak', size = 20)
plt.legend()

plt.subplot(426)
plt.xlim(0,400)
plt.ylim(-0.05,1.1)
plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), label = 'p3ht without first peak', c = 'dodgerblue')
plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),label = 'p3th without predict first,second peak', c = 'orange')
plt.title( 'p3th without predict first,second peak', size = 20)
plt.legend()

# plt.subplot(427)
# plt.xlim(0,400)
# plt.ylim(-0.05,1.1)
# plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
# # plt.plot(test_result[0].reshape(401,)- y(new_predict[1][0],new_predict[2][0],new_predict2[3][0],x), label = 'p3ht without right first peak', c = 'dodgerblue')
# plt.plot(test_result2[0].reshape(401,)- y(new_predict2[1][0],new_predict2[2][0],new_predict3[3][0],x),label = 'p3th without predict first,second right peak', c = 'orange')
# plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'third right real',color = 'purple', linewidth = 2)
# plt.plot(t,y(new_predict3[1][0],new_predict3[2][0],new_predict3[3][0],x),label = 'third right predict', c = 'blue')
# plt.title('compare third real, predict right peak', size = 20)
# plt.legend()

# plt.subplot(428)
# plt.xlim(0,400)
# plt.ylim(-0.05,1.1)
# plt.plot(test_result[0],label = 'total p3ht' ,c = 'black')
# plt.plot(test_result[0].reshape(401,)- y(new_predict[1][0],new_predict[2][0],new_predict2[3][0],x), label = 'p3ht without right first peak', c = 'dodgerblue')
# plt.plot(test_result2[0].reshape(401,)- y(new_predict2[1][0],new_predict2[2][0],new_predict3[3][0],x),label = 'p3th without predict first,second right peak', c = 'orange')
# plt.plot(test_result3[0].reshape(401,)- y(new_predict3[1][0],new_predict3[2][0],new_predict3[3][0],x),label = 'p3th without predict first,second,thrid right peak', c = 'royalblue')
# plt.title( 'p3th without predict first,second,third right peak', size = 20)
# plt.legend()


# In[1036]:


plt.figure(figsize = (10,5))
plt.plot(t,y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x), c = 'blue')
plt.plot(t,y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x), c = 'blue', label = 'predict')
# plt.plot(x,y(new_predict3[1][0],new_predict3[2][0],new_predict3[3][0],x),label = 'predict', c = 'blue')

plt.plot(t, (peak1[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2, label = 'real')
plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', linewidth = 2)
# plt.plot(x, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'real',color = 'black', linewidth = 2)
plt.plot(test_result[0], c = 'black', label = 'p3ht')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




