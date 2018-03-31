import jieba
import xlrd
import numpy as np
import random

def read_txt():
    data_pos = xlrd.open_workbook('pos.xls')
    data_neg = xlrd.open_workbook('neg.xls')
    table_pos = data_pos.sheets()[0]
    table_neg = data_neg.sheets()[0]
    col_value_pos=table_pos.col_values(0)
    col_value_neg=table_neg.col_values(0)

    return (col_value_pos,col_value_neg)

f = open('dic.txt','r')
a = f.read()
dic = eval(a)
f.close()

pos=[1.0]
neg=[0.0]
pos_list=[]
neg_list=[]
pos_list,neg_list=read_txt()
number_pos=[]
number_neg=[]
number_pos_label=[]
number_neg_label=[]
max_number=200

for line in pos_list:
    seg_list=jieba.cut(line,cut_all=False,HMM=True)
    tmp=[]
    tmp_number=[]
    number=0
    for word in seg_list:
        number=number+1
        tmp.append(word)
    if number<=max_number:
        for index in tmp:
            if index in dic:
                tmp_number.append(dic[index])
            else:
                tmp_number.append(0.0)
        cur_len=len(tmp_number)
        for i in range(max_number-cur_len):
            tmp_number.append(0.0)
        number_pos.append(tmp_number)
        number_pos_label.append(pos)

for line in neg_list:
    seg_list=jieba.cut(line,cut_all=False,HMM=True)
    tmp=[]
    tmp_number=[]
    number=0
    for word in seg_list:
        number=number+1
        tmp.append(word)
    if number<=max_number:
        for index in tmp:
            if index in dic:
                tmp_number.append(dic[index])
            else:
                tmp_number.append(0.0)
        cur_len=len(tmp_number)
        for i in range(max_number-cur_len):
            tmp_number.append(0.0)
        number_neg.append(tmp_number)
        number_neg_label.append(neg)

all_data=number_pos+number_neg
all_label=number_pos_label+number_neg_label

pos_neg=list(zip(all_data,all_label))
random.shuffle(pos_neg)
all_data[:],all_label[:]=zip(*pos_neg)
all_data_numpy=np.array(all_data)
all_label_numpy=np.array(all_label)
np.save('all_data_one.npy',all_data_numpy)
np.save('all_label_one.npy',all_label_numpy)


all_data_numpy=np.array(all_data)
all_label_numpy=np.array(all_label)
print(all_data_numpy.shape)
print(all_label_numpy.shape)




