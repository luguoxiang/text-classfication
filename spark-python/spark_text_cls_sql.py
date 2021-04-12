from pyspark import SparkContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from functools import partial
import math
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
spark = SparkSession(sc)
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
print("Category count: %d" % len(cls_index_map))
doc_cls_map = sc.broadcast(doc_cls_map)

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
    od = {}
    for (word_id, count) in x[1]:
        od[word_id] = count

    indexes = []
    values = []

    for word_id in sorted(od.keys()):
        indexes.append(word_id)
        values.append(od[word_id])
    doc_id = x[0]
    return LabeledPoint(doc_cls_map.value[doc_id], SparseVector(word_count.value, indexes, values))

word_count = None
def transform(files, word_filtered, tf_idf):
    global word_count
    word_doc = files.flatMap(get_words).toDF(["word","doc_id"])
    word_doc_count = word_doc.groupBy("word","doc_id").count().cache()
    
    doc_count = files.count()
    if not word_filtered:
        word_idf = word_doc_count.withColumn("one", lit(1)).groupBy("word").sum("one").withColumnRenamed("SUM(one)", "idf")
        word_filtered = word_idf.filter("idf >= 5 and idf <= %f" % (0.3 * doc_count ))
        word_filtered = word_filtered.rdd.zipWithIndex()
        if tf_idf:
            mapFn = lambda x: (x[0][0], x[1], math.log(doc_count / (x[0][1]+ 0.01)))
            word_filtered = word_filtered.map(mapFn)
        else:
            word_filtered = word_filtered.map(lambda x: (x[0][0], x[1], 1))
        word_count = sc.broadcast(word_filtered.count())
        word_filtered = word_filtered.toDF(["word", "word_id", "w"]).cache()
    else:
        assert word_count
    word_doc_count = word_doc_count.join(word_filtered, word_doc_count.word ==  word_filtered.word)
    doc_word_weight = word_doc_count.withColumn("weight", (col("count") * col("w"))).select("doc_id", "word_id", "weight")
    doc_word_weight = doc_word_weight.rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()

    return doc_word_weight.map(partial(to_word_vector, word_count)), word_filtered


filesRDD = sc.parallelize(files)
training, test = filesRDD.randomSplit([0.6, 0.4])
training, word_filtered = transform(training, None, False)
test, _ = transform(test, word_filtered, False)
model = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
print("####################")
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))
print("####################")

