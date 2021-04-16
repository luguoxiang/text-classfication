package com.guoxiang.hdfs 
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path, LocatedFileStatus,FileStatus,FSDataInputStream}
import java.net.URI
import java.io.{InputStream, BufferedReader, InputStreamReader}
import scala.collection.immutable.Stream
import opennlp.tools.stemmer.PorterStemmer
import com.huaban.analysis.jieba.JiebaSegmenter
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, SparseVector}
import org.apache.spark.ml.classification.{
        NaiveBayes, 
        LinearSVC,
        OneVsRest,
        LogisticRegression,
        DecisionTreeClassifier,
        RandomForestClassifier, 
        RandomForestClassificationModel,
}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import scala.collection.JavaConverters._
import TextClassifier._

import java.lang.Math

@SerialVersionUID(1000L)
class TextClassifier extends Serializable {
    var cn_stop_words : Broadcast[Set[String]] = null
    var en_stop_words : Broadcast[Set[String]] = null

    //doc_cls_map(doc_id) => cls_id
    var doc_cls_map : Broadcast[Array[Int]] = null

    def readStopWords(path: String) = {
        var input : InputStream = null
        var reader : BufferedReader = null
        try {
            input = getClass.getResourceAsStream(path);
            reader = new BufferedReader(new InputStreamReader(input, DICT_ENCODING))
            def readLines = Stream.continually( reader.readLine)
            readLines.takeWhile(_ != null).filter(_.strip().length() !=0)
                .toSet
        }finally {
            if(reader != null) {
                reader.close();
            }
            if(input != null) {
                input.close()
            }
        }

    } 

    def readWords(path : Path, 
                doc_id: Int, 
                hdfs: FileSystem) = {
        var input : FSDataInputStream = null
        try {
            input = hdfs.open(path)
            "[a-zA-Z]+|[\u4e00-\u9fa5]+".r
                .findAllIn(new String(input.readAllBytes(), TEXT_ENCODING))
                .flatMap(x => if(x(0) >= 0x4e00 && x(0) <= 0x9fa5) {
                    segmenter.process(x, JiebaSegmenter.SegMode.SEARCH)
                        .asScala
                        .map(_.word)
                        .filter(!cn_stop_words.value.contains(_))
                } else {
                    Array(stemmer.stem(x.toLowerCase()))
                        .filter(!en_stop_words.value.contains(_))
                }).map((_, doc_id)) 
        } finally {
            if(input != null) {
                input.close();
            }
        }
    }
    def connectHDFS() = {
        FileSystem.get(new URI(HDFS_URL), new Configuration(), HDFS_USER) 
    }

    def prepareFeatures(files: RDD[(Path, Int)],
            word_total_count: RDD[(String, (Long, Int))],
            termWeight : TextClassifier.TermWeight) = {

        val doc_count = files.count()
        val word_doc_tf = files.mapPartitions(iterator => {
            val hdfs = connectHDFS();
            iterator.flatMap{case (path, doc_id) => 
                readWords(path, doc_id, hdfs)
            }
        }).map(x => (x, 1)).reduceByKey((x, y) => x + y)

        var word_dict = word_total_count
        if (word_dict == null) {
            word_dict = word_doc_tf
                .map{case ((word, doc_id), count)=> (word, 1)}
                .reduceByKey((x, y) => x + y)
                .filter{case (word, df) => df >=5 && df <= 0.3 * doc_count}
                .zipWithIndex
                .map{
                    case ((word, docf), word_id) => (word, (word_id, docf))
                }
            word_dict = word_dict.cache()
        }

        val dict_size = word_dict.count().toInt
        val features = word_doc_tf
            .map{case ((word, doc_id), termf) => (word, (doc_id, termf))}
            .join(word_dict)
            .map{case (word, ((doc_id, termf), (word_id, docf))) 
                => (doc_id, (word_id, termWeight.compute(termf, docf, doc_count)))}
            .groupByKey()
            .map{case (doc_id, wordList) => {
                val sortedWords = wordList.iterator.toArray.sortBy{case (word_id, _) => word_id}

                (doc_cls_map.value(doc_id), 
                    new SparseVector(dict_size, 
                        sortedWords.map{case (word_id, weight) => word_id.toInt}.toArray, 
                        sortedWords.map{case (word_id, weight) => weight}.toArray))
            }}.cache()
        (features, word_dict)
    }

    def classify(classification_fn: (DataFrame, DataFrame) => DataFrame,
            termWeight : TextClassifier.TermWeight) {
        val hdfs = connectHDFS();

        val classes = hdfs.listStatus(new Path(TEXT_BASE_DIR))
            .filter(_.isDir)
            .map(_.getPath).zipWithIndex

        val files = classes.flatMap{case (f, cls_index) =>
                hdfs.listStatus(f).filter(!_.isDir).map(x => (x.getPath(), cls_index))
            }

        val documents = files.map{case (path, _) => path}.zipWithIndex

        printf("Document count: %d\n", files.size)
        printf("Category count: %d\n", classes.size)
      
        printf("HADOOP_CONF_DIR=%s\n", System.getenv("HADOOP_CONF_DIR"))
        printf("YARN_CONF_DIR=%s\n", System.getenv("YARN_CONF_DIR"))

        val spark =  SparkSession
            .builder()
            .appName("TextClassification")
            .getOrCreate()
        val sc: SparkContext = spark.sparkContext

        doc_cls_map = sc.broadcast(
            files.map{case (_, cls_index) => cls_index}.toArray)

        val documentsRDD = sc.parallelize(documents)
        
        println("loading stop words...")
        cn_stop_words = sc.broadcast(readStopWords("/cn_stop.txt"))
        en_stop_words = sc.broadcast(readStopWords("/en_stop.txt"))
        printf("english stop words: %d\n", en_stop_words.value.size)
        printf("chinese stop words: %d\n", cn_stop_words.value.size)

        val Array(training, test) = documentsRDD.randomSplit(Array(0.6, 0.4))

        val (trainingFeatures, word_total_count) = prepareFeatures(
            training, null, termWeight)

        val (testFeatures, _) = prepareFeatures(
            test, word_total_count, termWeight)

        println("Dictionary size %d".format(word_total_count.count()))

        val trainingFeaturesDF =  spark
            .createDataFrame(trainingFeatures)
            .toDF(COLUMN_LABEL, COLUMN_FEATURES)
        val testFeaturesDF =  spark
            .createDataFrame(testFeatures)
            .toDF(COLUMN_LABEL, COLUMN_FEATURES)

        val predictions = classification_fn(
            trainingFeaturesDF, testFeaturesDF)
        predictions.show()

        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(COLUMN_LABEL)
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test set accuracy = $accuracy")
    }
}

object TextClassifier {
    val stemmer = new PorterStemmer
    val segmenter = new JiebaSegmenter

    val TEXT_BASE_DIR = "/news_text"
    val HDFS_URL = "hdfs://hadoop-master:9000"
    val HDFS_USER = "hduser"
    val DICT_ENCODING = "utf-8"
    val TEXT_ENCODING = "gb2312"
    val COLUMN_FEATURES = "features"
    val COLUMN_LABEL = "label"

    trait TermWeight extends Serializable{
        def compute(tf: Int, df : Int, doc_count: Long) : Double
    }
    class TfTermWeight extends TermWeight{
        def compute(tf: Int, docf : Int, doc_count: Long) = {
            tf.toDouble
        }
    }
    class TfIdfTermWeight extends TermWeight{
        def compute(tf: Int, docf : Int, doc_count: Long) = {
            tf * Math.log((doc_count.toDouble + 1)/ (docf + 1))
        }
    }

    def main(args: Array[String]) {
        val randomForestClassifier = new RandomForestClassifier() 
                .setMaxBins(16)
                .setMaxDepth(30)
                .setNumTrees(200)

        val decisionTreeClassifier = new DecisionTreeClassifier()
                .setMaxBins(16)
                .setMaxDepth(20)

        val naiveBayes = new NaiveBayes()
        
        val linearSVC = new OneVsRest().setClassifier(
                new LinearSVC()
                .setMaxIter(100)
                .setRegParam(0.1)
            )

            
        //var termWeight = new TfTermWeight()
        var termWeight = new TfIdfTermWeight()

        new TextClassifier().classify(
            (training, test) => linearSVC.fit(training).transform(test),
            termWeight)
    }
}

