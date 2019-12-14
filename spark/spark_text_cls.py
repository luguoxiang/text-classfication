from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from functools import partial
import boto3
import nltk
import jieba
import itertools

OTHER_CH = 0
CN_CH = 1
EN_CH = 2
def ch_type(ch):
    if ch>='a' and ch<='z':
        return EN_CH
    if ch>='A' and ch <='Z':
        return EN_CH
    if ch >='\u4e00' and ch <='\u9fa5':
        return CN_CH
    return OTHER_CH


sc = SparkContext(appName="TestApp")
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('text-cls-data')

all_objects = list(my_bucket.objects.all())
cls_index_map = {}
doc_cls_map = [None] * len(all_objects)
files = []

for index in range(len(all_objects)):
    name = all_objects[index].key
    name_items = name.split("/")
    if len(name_items) != 3:
        continue
    cls = name_items[1]

    if cls not in cls_index_map:
        cls_index_map[cls]=len(cls_index_map)
    doc_cls_map[index] = cls_index_map[cls]
    files.append((name, index))

print("Document count: %d" % len(files))
print("Category count: " + len(cls_index_map))
doc_cls_map = sc.broadcast(doc_cls_map)

files = sc.parallelize(files)

print("loading stop words...")
stop_obj = s3.Object("text-cls-data","stop.txt")
stop_body = stop_obj.get()['Body'].read()
cn_stop_words={word for word in stop_body.decode("utf-8").split('\n') if word.strip()!=''}
cn_stop_words=sc.broadcast(cn_stop_words)
nltk.download('stopwords')
en_stop_words = set(nltk.corpus.stopwords.words('english'))
en_stop_words.add('nsbp') #skip &nsbp;
en_stop_words=sc.broadcast(en_stop_words)
stemmer = sc.broadcast(nltk.stem.PorterStemmer())

def get_words(file_info):
    file, doc_id = file_info
    s3 = boto3.resource('s3')
        
    obj = s3.Object("text-cls-data", file)
    body = obj.get()['Body'].read()
    document=body.decode("utf-8")
    word_list = []
    for current_type,segment in itertools.groupby(document, ch_type):
            segment = ''.join(segment)
            if current_type == EN_CH and len(segment)> 2:
                w = segment.lower()
                if w not in en_stop_words.value:
                    w = stemmer.value.stem(w)
                    word_list.append((w, doc_id))
            elif current_type == CN_CH:
                for w in jieba.cut(segment):
                    if w not in cn_stop_words.value:
                        word_list.append((w, doc_id))
    return word_list

def to_word_vector(word_count, x):
    doc_id = x[0]
    vector = [0] * word_count.value
    for (word_id, count) in x[1]:
        vector[word_id] = count
    vector.append(doc_cls_map.value[doc_id])
    return LabeledPoint(doc_cls_map.value[doc_id], vector)

word_count = None
def transform(lines, word_filtered):
    global word_count
    word_doc_count = lines.flatMap(get_words).map(lambda word_doc:(word_doc,1)).reduceByKey(lambda a, b: a + b).cache()
    if not word_filtered:
        word_idf = word_doc_count.map(lambda x: (x[0][0], 1)).reduceByKey(lambda a, b: a + b)
        #word_tf = word_doc_count.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda a, b: a + b)
        #word_filtered = word_idf.filter(lambda x:x[1] >=5 and x[1] <= 0.3 * len(files))
        word_filtered = word_idf
        word_filtered = word_filtered.zipWithIndex().map(lambda x: (x[0][0], x[1])).cache()
        word_count = sc.broadcast(word_filtered.count())
    else:
        assert word_count
    word_doc_count = word_doc_count.map(lambda x:(x[0][0],(x[0][1], x[1]))).join(word_filtered).map(lambda x:(x[1][1],x[1][0][0],x[1][0][1]))
    doc_word_count = word_doc_count.map(lambda x: (x[1], (x[0], x[2]))).groupByKey()

    #word_list = [None] * word_count.value
    #for (word, word_id) in word_filtered.collect():
    #    word_list[word_id] = word
    #word_list.append("Category")
    return doc_word_count.map(partial(to_word_vector, word_count)), word_filtered
    
    
my_bucket = s3.Bucket('text-cls-data')

training, test = files.randomSplit([0.6, 0.4])
training, word_filtered = transform(training, None)
test, _ = transform(test, word_filtered)
model = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
print("####################")
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))
print("####################")

