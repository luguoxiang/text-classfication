import nltk
import sys
import os
import time
import itertools
import jieba
import numpy
import scipy.sparse

TEXT_ENCODING="utf-8"
print("loading stop words...")
nltk.download('stopwords')
en_stop_words = set(nltk.corpus.stopwords.words('english'))
en_stop_words.add('nsbp') #skip &nsbp;

with open("dict/stop.txt", 'r', encoding='utf-8') as stop_file:
    cn_stop_words={word for word in stop_file.read().split('\n') if word.strip()!=''}

print("english stopwords: {}".format(len(en_stop_words)))
print("chinese stopwords: {}".format(len(cn_stop_words)))

OTHER_CH = 0
CN_CH = 1
EN_CH = 2
global_stemmer = nltk.stem.PorterStemmer()

def ch_type(ch):
    if ch>='a' and ch<='z':
        return EN_CH
    if ch>='A' and ch <='Z':
        return EN_CH
    if ch >='\u4e00' and ch <='\u9fa5':
        return CN_CH
    return OTHER_CH

def get_word_list(document):
	word_list = []
	for current_type,segment in itertools.groupby(document.read(), ch_type):
		segment = ''.join(segment)
		if current_type == EN_CH and len(segment)> 2:
			w = segment.lower()
			if w not in en_stop_words:
				w = global_stemmer.stem(w)
				word_list.append(w) 
		elif current_type == CN_CH:
			for w in jieba.cut(segment):
				if w not in cn_stop_words:
					word_list.append(w) 
	return word_list

gloabl_word_index = {}
global_word_count = 0

def get_word_index(word):
    global global_word_count
    global gloabl_word_index

    if word in gloabl_word_index:
        return gloabl_word_index[word]
    else:
        gloabl_word_index[word] = global_word_count
        global_word_count = global_word_count + 1
        return global_word_count -1

root_dir=sys.argv[1]
print("loading data from {}...".format(root_dir))

cat_list = os.listdir(root_dir)
doc_cat = []
rows, cols, data = [],[],[]
        
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
		with open(sample_path,'r', encoding=TEXT_ENCODING, errors='ignore') as sample_file:
			word_list=get_word_list(sample_file)
			for word,v in itertools.groupby(sorted(word_list)):
				word_index=get_word_index(word)  
				if word_index <0: continue

				rows.append(doc_index)
				cols.append(word_index)
				data.append(sum(1 for i in v))
		process_time = process_time + time.time() - start_time

	print("loading category: {}({}/{}), time: {}".format(
            cat_list[cat_index], cat_index +1, len(cat_list), process_time))

doc_word_matrix=scipy.sparse.coo_matrix((data, (rows,cols)))
target_value=numpy.array(doc_cat)

print("data shape:{}, target shape:{}".format(doc_word_matrix.shape, target_value.shape))
			
scipy.sparse.save_npz("data_tf", doc_word_matrix)
numpy.savez_compressed("target", target=target_value)

