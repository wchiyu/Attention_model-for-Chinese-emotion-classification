#coding:utf-8
import  jieba
import xlrd

def read_txt():
    data_pos = xlrd.open_workbook('pos.xls')
    data_neg = xlrd.open_workbook('neg.xls')
    table_pos = data_pos.sheets()[0]
    table_neg = data_neg.sheets()[0]
    col_value_pos=table_pos.col_values(0)
    col_value_neg=table_neg.col_values(0)
    return (col_value_pos,col_value_neg)

pos_list=[]
neg_list=[]
pos_list,neg_list=read_txt()

pos_seg=[]
neg_seg=[]
dic={}
dic_word=[]
dic_f={}
dic_t={}

for line in pos_list:
    seg_list=jieba.cut(line,cut_all=False,HMM=True)
    tmp=[]
    for word in seg_list:
        tmp.append(word)
        if word in dic:
            dic[word]=dic[word]+1
        else:
            dic[word]=1
    pos_seg.append(tmp)

for line in neg_list:
    seg_list=jieba.cut(line,cut_all=False,HMM=True)
    tmp=[]
    for word in seg_list:
        tmp.append(word)
        if word in dic:
            dic[word]=dic[word]+1
        else:
            dic[word]=1
    neg_seg.append(tmp)

#print(type(sorted(dic.items(),key=lambda dic:dic[1])))

for key in dic.keys():
    if dic[key]>=5:
        dic_t[key]=dic[key]

dic_word=sorted(dic_t.items(),key=lambda dic_t:dic_t[1],reverse=True)

#print(type(dic_word))
#for i in dic_word:
#    print(i)

number=1.0
for i in dic_word:
    dic_f[str(i[0])]=number
    number=number+1


f = open('dic.txt','w')
f.write(str(dic_f))
f.close()





