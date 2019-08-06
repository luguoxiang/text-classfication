# -*- coding: utf-8 -*-
import json
import jieba
import itertools
import os
import sys
import string
import gensim
import logging
import concurrent.futures
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

OTHER_CH=0
CN_CH=1
EN_CH=2
NUM_CH=3
EMPTY_CH=4
SPLIT_CH=5

type_dict = [OTHER_CH] * 256
for c in string.digits:
    type_dict[ord(c)] = NUM_CH
for c in string.ascii_uppercase + string.ascii_lowercase:
    type_dict[ord(c)] = EN_CH
type_dict[ord(' ')] = EMPTY_CH
type_dict[ord('\t')] = EMPTY_CH
type_dict[ord('\r')] = EMPTY_CH
type_dict[ord(',')] = SPLIT_CH
type_dict[ord(';')] = SPLIT_CH
type_dict[ord('!')] = SPLIT_CH
type_dict[ord('?')] = SPLIT_CH

def ch_type(ch):
    index = ord(ch)
    if index < 256:
        return type_dict[index]
    if ch >='\u4e00' and ch <='\u9fa5':
        return CN_CH
    if ch == '。' or ch == '；' or ch=='，' or ch == '！' or ch == '？':
        return SPLIT_CH
    return OTHER_CH

def get_word_list(document_path):
    result = []
    with open(document_path,'r', encoding='utf-8', errors='ignore') as document:
        for line in document:
            data = json.loads(line)
            sentences = data['text'].split("\n")
            sentences.append(data['title'])
       
            for text in sentences:
                word_list = []
                has_cn = False
                for current_type,segment in itertools.groupby(text, ch_type):
                    if current_type == EN_CH:
                        word_list.append("(english)")
                    elif current_type == NUM_CH:
                        word_list.append("(number)") 
                    elif current_type == CN_CH:        
                        segment = ''.join(segment)   
                        has_cn = True  
                        for w in jieba.cut(segment):
                            word_list.append(w) 
                    elif current_type == SPLIT_CH: 
                        if len(word_list) > 3 and has_cn:
                            result.append(word_list)
                        word_list=[]
                        has_cn = False
                if len(word_list) > 3 and has_cn:
                    result.append(word_list)
        return result

def get_documents(root_dir):
    result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for cat_dir in os.listdir(root_dir): # iterate category  
            cat_path = os.path.join(root_dir, cat_dir)  
            if not os.path.isdir(cat_path):
                continue    
            print("Searching {}".format(cat_dir))
            for sample_file in os.listdir(cat_path): # iterate document under category
                sample_path = os.path.join(cat_path, sample_file)          
                if os.path.isdir(sample_path): continue
                
                future = executor.submit(get_word_list, sample_path)
                result.append(future)
              
        sentences = []
        done = 0
        for future in concurrent.futures.as_completed(result):
            doc = future.result()
            done = done +  1
            sentences = sentences + doc
            print("Processed {}/Total {}".format(done, len(result)))
        return sentences
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root",  default="")
    ns = parser.parse_args()
    print("loading data from {}...".format(ns.root))
    sentences = get_documents(ns.root)
    print("Traing Word2Vec model...")
    model = gensim.models.Word2Vec(
            sentences,
            size=300,
            window=10,
            min_count=2,
            workers=10,
            iter=10)
    print("Saving...")
    model.save('word2vec.model')
    print("Done")
