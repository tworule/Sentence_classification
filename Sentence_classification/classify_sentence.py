import json
import pandas
import math
import numpy
from pandas import Series, DataFrame

import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from nltk.corpus import stopwords


# LOAD WORD2VEC MODEL & STOP WORD
model = Word2Vec.load('./gensim_model_skip_gram')     #Yelp Review Skip Gram으로 돌린 모델
stop_words = set(stopwords.words('english'))


# CREATE DISTANCE MATRIX
food_row = model.most_similar(positive=['food'],topn=10000**10000)
service_row = model.most_similar(positive=['service'],topn=10000**10000)
ambience_row = model.most_similar(positive=['ambience'],topn=10000**10000)
value_row = model.most_similar(positive=['value'],topn=10000**10000)

col_name_list = []
col_value_list = []
for i in food_row:
    col_name_list.append(i[0])
    col_value_list.append(distance.euclidean(model['food'],model[i[0]]))
food_df = pandas.DataFrame(columns = col_name_list)
food_df.loc[0] = col_value_list

col_name_list = []
col_value_list = []
for i in service_row:
    col_name_list.append(i[0])
    col_value_list.append(distance.euclidean(model['service'],model[i[0]]))
service_df = pandas.DataFrame(columns = col_name_list)
service_df.loc[0] = col_value_list

col_name_list = []
col_value_list = []
for i in ambience_row:
    col_name_list.append(i[0])
    col_value_list.append(distance.euclidean(model['ambience'],model[i[0]]))
ambience_df = pandas.DataFrame(columns = col_name_list)
ambience_df.loc[0] = col_value_list

col_name_list = []
col_value_list = []
for i in value_row:
    col_name_list.append(i[0])
    col_value_list.append(distance.euclidean(model['value'],model[i[0]]))
value_df = pandas.DataFrame(columns = col_name_list)
value_df.loc[0] = col_value_list

distance_matrix = pandas.concat([food_df,service_df,ambience_df,value_df])
distance_matrix.index = ['food','service','ambience','value']
distance_matrix = distance_matrix.fillna(0)


# CREATE WEIGHT MATRIX
dis_matrix = distance_matrix
dis_values = dis_matrix.values
dis_values2 = dis_matrix.values
dis_values2 = list(dis_values2)

w_list = []
for i in dis_values:
    w_list2 = []
    for j in i:
        w_list2.append(math.exp(-(j**2)/2))
    w_list.append(w_list2)

dis_values2[0] = w_list[0]
dis_values2[1] = w_list[1]
dis_values2[2] = w_list[2]
dis_values2[3] = w_list[3]

final_col_list = list(distance_matrix.columns.values)
weight_final_df = pandas.DataFrame(columns = final_col_list)

weight_final_df.loc[0] = w_list[0]
weight_final_df.loc[1] = w_list[1]
weight_final_df.loc[2] = w_list[2]
weight_final_df.loc[3] = w_list[3]
weight_final_df.index = ['food','service','ambience','value']


# 가중치의 편차가 0.06 이하인 단어들은 가중치 * 0.1
def std(string):
    x1 = weight_final_df[string][0]
    x2 = weight_final_df[string][1]
    x3 = weight_final_df[string][2]
    x4 = weight_final_df[string][3]
    return(numpy.std([x1, x2, x3, x4]))

col_list = list(weight_final_df)

for i in col_list:
    if std(str(i)) <= 0.06:
        weight_final_df[i][0] = weight_final_df[i][0] * 0.1
        weight_final_df[i][1] = weight_final_df[i][1] * 0.1
        weight_final_df[i][2] = weight_final_df[i][2] * 0.1
        weight_final_df[i][3] = weight_final_df[i][3] * 0.1


# TEST
while True:
    print('----------------------------------------------------------------------------------')
    test_str = input('문장 입력 (c는 종료) : ')
    print('----------------------------------------------------------------------------------')
    if test_str == 'c':
        break

    try:
        # create TDM(term-document matrix)
        TDM_col_list = list(weight_final_df.columns.values)
        TDM_df = pandas.DataFrame(columns = TDM_col_list)
        TDM_df.loc[0] = 0 * len(TDM_col_list)

        input_str = test_str
        input_str = input_str.lower()
        input_str = input_str.replace(".","")
        input_str = input_str.replace("!","")
        input_str_list = input_str.split(' ')

        for i in input_str_list:
            if not i in stop_words:
                try:
                    TDM_df[str(i)][0] = 1
                except KeyError:
                    continue
            else:
                continue

        TDM_df = TDM_df.T
        
        # 가중치 행렬, TDM 내적하기.(weight_matrix * TDM_df)
        score = weight_final_df.dot(TDM_df)
        print('-------------------------------------------------')
        score_list = []
        score_list.append(('food',float(score[0][0])))
        score_list.append(('service',float(score[0][1])))
        score_list.append(('ambience',float(score[0][2])))
        score_list.append(('value',float(score[0][3])))
        xx = sorted(score_list, key = lambda score: score[1], reverse = True)
        print('first : ', xx[0])
        print('second : ', xx[1])
        print('third : ', xx[2])
        print('fourth : ', xx[3])
        print('-------------------------------------------------')
    except KeyError:
        print('다른 문장 입력')
        continue
