"""
@author: 成昊
@desc: 对 dataframe 进行分词处理, 生成 tfidf 模型用于匹配
"""
import os
import sys
import pickle
import numpy as np
from config import *
from trie import TrieTree
from sklearn.feature_extraction.text import TfidfVectorizer


def text2tfidf(text):
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        tfidf = tv.fit_transform(text)
    except ValueError:
        print(text)
        assert False
    return tfidf, tv


def run():
    print('Preprocess start')
    with open(DATA_PATH, 'rb') as data_file:
        jds = pickle.load(data_file)
    jds['combined'] = jds['job_title'] + jds['job_description']
    del jds['job_title']
    del jds['job_description']
    print('JD data loaded')

    print('Cutting sentence...')
    trie = TrieTree(SKILL_PATH)

    cut_jd = []
    for idx, row in jds.iterrows():
        cut = trie.cut(row['combined'])
        cut_jd.append(' '.join(cut))
        if idx % 1000 == 0 or idx == len(jds)-1:
            print('\rProcessing %.2f%%' % (100*(idx+1)/len(jds)), end='')
    del jds['combined']
    print('Done!')

    print('Converting to TF-IDF...')
    tfidf, tv = text2tfidf(cut_jd)
    print('Done with tfidf size: ', tfidf.shape)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump((tfidf, tv), f)
    print('All done!')


if __name__ == '__main__':
    run()
