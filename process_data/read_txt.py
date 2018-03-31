#coding:utf-8
import xlrd

def read_txt():
    data_pos = xlrd.open_workbook('pos.xls')
    data_neg = xlrd.open_workbook('neg.xls')
    table_pos = data_pos.sheets()[0]
    table_neg = data_neg.sheets()[0]
    col_value_pos=table_pos.col_values(0)
    col_value_neg=table_neg.col_values(0)

    return (col_value_pos,col_value_neg)


