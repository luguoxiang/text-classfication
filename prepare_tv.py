import jieba
import scipy.sparse 
import time
import itertools
import os
import sys
import nltk.stem 
import string
import numpy

OTHER_CH=0
CN_CH=1
EN_CH=2
NUM_CH=3
EMPTY_CH=4

type_dict = [OTHER_CH] * 256
for c in string.digits + ".,":
    type_dict[ord(c)] = NUM_CH
for c in string.ascii_uppercase + string.ascii_lowercase + "'":
    type_dict[ord(c)] = EN_CH
type_dict[ord(' ')] = EMPTY_CH
type_dict[ord('\t')] = EMPTY_CH

def ch_type(ch):
    index = ord(ch)
    if index < 256:
        return type_dict[index]
    if ch >='\u4e00' and ch <='\u9fa5':
        return CN_CH
    return OTHER_CH


WORD_INDEX_PAD=0
WORD_INDEX_OTHER=1
WORD_INDEX_NUMBER=2

global_word_count = 3
gloabl_word_index = {}
global_stemmer=nltk.stem.PorterStemmer()

def get_word_index(word):
    global global_word_count
    global gloabl_word_index

    if word in gloabl_word_index:
        index = gloabl_word_index[word]
        return index
    else:
        gloabl_word_index[word] = global_word_count
        global_word_count = global_word_count + 1
        return global_word_count -1

root_dir=sys.argv[1]
print("loading data from {}...".format(root_dir))

cat_list = os.listdir(root_dir)
doc_cat = []
doc_word_list = []

def get_word_list(document):
    word_list = []
    for current_type,segment in itertools.groupby(document.read(), ch_type):
        segment = ''.join(segment)
        if current_type == EMPTY_CH:
           continue
        elif current_type == EN_CH and len(segment) > 1:
            w = global_stemmer.stem(segment.lower())
            word_list.append(get_word_index(w))
        elif current_type == NUM_CH:
            index = WORD_INDEX_OTHER
            for ch in segment:
                if ch>='0' and ch <='9':
                    index = WORD_INDEX_NUMBER
                    break
            word_list.append(index)
        elif current_type == CN_CH:
            for w in jieba.cut(segment):
                word_list.append(get_word_index(w))
    return word_list

for cat_index in range(len(cat_list)): # iterate category
    cat_dir = cat_list[cat_index]
    cat_path=os.path.join(root_dir, cat_dir)
    if not os.path.isdir(cat_path): continue
        
    process_time = 0.0
    for sample_file in os.listdir(cat_path): # iterate document under category
        sample_path = os.path.join(cat_path, sample_file)
        if os.path.isdir(sample_path): continue

        doc_index = len(doc_cat)
        doc_cat.append(cat_index)

        start_time = time.time()
        with open(sample_path,'r', encoding='gb18030', errors='ignore') as sample_file:
            word_list=get_word_list(sample_file)
            doc_word_list.append(numpy.array(word_list))
            
        process_time = process_time + time.time() - start_time

    print("loading category: {}({}/{}), time: {}".format(
        cat_list[cat_index], cat_index +1, len(cat_list), process_time))

target=numpy.array(doc_cat)

doc_len = [len(doc) for doc in doc_word_list]
doc_num = len(doc_len)
max_len = sorted(doc_len) [int(doc_num * 0.99)]

doc_word_list = [doc[0:max_len] if doc.shape[0] > max_len else doc for doc in doc_word_list]

data = numpy.stack([numpy.pad(doc, (0, max_len - doc.shape[0]), 'constant') for doc in doc_word_list])

print("data: {}, target: {}".format(data.shape, target.shape))
numpy.savez_compressed("data_tv", data=data, target=target)
