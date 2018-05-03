from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors as word2vec
import pandas
from pandas import Series, DataFrame
from nltk.corpus import stopwords
import json
import sys
sys.stdout.flush()


# PREPROCESSING
def basic_preprocessing(input_list):
    stop_words = set(stopwords.words('english'))
    text_list2 = [x.lower() for x in input_list]
    text_list3 = [x.replace(".","") for x in text_list2]
    text_list3 = [x.replace("!","") for x in text_list3]
    text_list3 = [x.replace("(","") for x in text_list3]
    text_list3 = [x.replace(")","") for x in text_list3]
    text_list3 = [x.replace("[","") for x in text_list3]
    text_list3 = [x.replace("]","") for x in text_list3]
    text_list3 = [x.replace("-","") for x in text_list3]
    text_list3 = [x.replace("\'","") for x in text_list3]
    text_list3 = [x.replace("\"","") for x in text_list3]
    text_list3 = [x.replace("@","") for x in text_list3]
    text_list3 = [x.replace("#","") for x in text_list3]
    text_list3 = [x.replace("^","") for x in text_list3]
    text_list3 = [x.replace(",","") for x in text_list3]
    text_list3 = [x.replace("?","") for x in text_list3]
    text_list3 = [x.replace("\n"," ") for x in text_list3]
    text_list3 = [x.replace(":"," ") for x in text_list3]
    text_list3 = [x.replace(";"," ") for x in text_list3]
    text_list4 = []
    for i in text_list3:
        text_list4.append(i.split(' '))

    text_list5 = []
    for i in text_list4:
        filtered_sentence = [w for w in i if not w in stop_words]
        text_list5.append(filtered_sentence)
    return(text_list5)


# WORD EMBEDDING
def word_embedding(input_list):
    print('SKIP_GRAM_word embedding', flush = True)
    model_skip_gram = Word2Vec(text_list4, size = 100, window = 5, min_count = 50, iter = 50, sg = 1)
    model_skip_gram.init_sims(replace=True)
    model_skip_gram.save('./embedding_model_SkipGram')
    return(model_skip_gram)