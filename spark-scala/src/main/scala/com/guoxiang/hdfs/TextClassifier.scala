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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.tree.{DecisionTree,RandomForest}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import scala.collection.JavaConverters._
import TextClassifier.{
        DecisionTreeClassifier, 
        NaiveBayesClassifier, 
        RandomForestClassifier, 
        Classifier
    }

import java.lang.Math

@SerialVersionUID(1000L)
class TextClassifier extends Serializable {

    val TEXT_ENCODING = "utf-8"
    var cn_stop_words : Broadcast[Set[String]] = null
    var en_stop_words : Broadcast[Set[String]] = null

    //doc_cls_map(doc_id) => cls_id
    var doc_cls_map : Broadcast[Array[Int]] = null

    def readStopWords(path: String) = {
        var input : InputStream = null
        var reader : BufferedReader = null
        try {
            input = getClass.getResourceAsStream(path);
            reader = new BufferedReader(new InputStreamReader(input, "utf-8"))
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
                    TextClassifier.segmenter.process(x, JiebaSegmenter.SegMode.SEARCH)
                        .asScala
                        .map(_.word)
                        .filter(!cn_stop_words.value.contains(_))
                } else {
                    Array(TextClassifier.stemmer.stem(x.toLowerCase()))
                        .filter(!en_stop_words.value.contains(_))
                }).map((_, doc_id)) 
        } finally {
            if(input != null) {
                input.close();
            }
        }
    }
    def connectHDFS() = {
        FileSystem.get(new URI("hdfs://ubuntu-master:9000"), new Configuration(), "hduser") 
    }

    def prepareFeatures(files: RDD[(Path, Int)],
            word_total_count: RDD[(String, (Long, Int))],
            classifier : Classifier) = {

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
                => (doc_id, (word_id, classifier.getTermWeight(termf, docf, doc_count)))}
            .groupByKey()
            .map{case (doc_id, wordList) => {
                val sortedWords = wordList.iterator.toArray.sortBy{case (word_id, _) => word_id}

                new LabeledPoint(doc_cls_map.value(doc_id), 
                    new SparseVector(dict_size, 
                        sortedWords.map{case (word_id, weight) => word_id.toInt}.toArray, 
                        sortedWords.map{case (word_id, weight) => weight}.toArray))
            }}
        (features, word_dict)
    }
    def ptest(path: Path) = {
        path
    }
    def classify(method : String) {
        val hdfs = connectHDFS();

        val classes = hdfs.listStatus(new Path("/news_text"))
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
        val conf = new SparkConf().setAppName("TextClassification")

        val sc = new SparkContext(conf);

        doc_cls_map = sc.broadcast(
            files.map{case (_, cls_index) => cls_index}.toArray)

        val documentsRDD = sc.parallelize(documents)
        
        println("loading stop words...")
        cn_stop_words = sc.broadcast(readStopWords("/cn_stop.txt"))
        en_stop_words = sc.broadcast(readStopWords("/en_stop.txt"))
        printf("english stop words: %d\n", en_stop_words.value.size)
        printf("chinese stop words: %d\n", cn_stop_words.value.size)

        val Array(training, test) = documentsRDD.randomSplit(Array(0.6, 0.4))
        var classifier : Classifier = null
        if (method.equals("random-forest")) {
            classifier = new RandomForestClassifier(classes.size)
        } else if (method.equals("decision-tree")) {
            classifier = new DecisionTreeClassifier(classes.size)
        } else {
            classifier = new NaiveBayesClassifier()
        }
        val (trainingFeatures, word_total_count) = prepareFeatures(
            training, null, classifier)

        val (testFeatures, _) = prepareFeatures(
            test, word_total_count, classifier)

        
        classifier.fit(trainingFeatures)
        val predictionAndLabel = testFeatures.map(
            p => (classifier.predict(p.features), p.label))
        println("####################")
        val accuracy = predictionAndLabel.filter{
            case (predicated, labled) => predicated == labled}.count().toDouble / testFeatures.count()
        printf("model accuracy %f\n", accuracy)
        println("####################")
    }
}

object TextClassifier {
    val stemmer = new PorterStemmer
    val segmenter = new JiebaSegmenter

    trait Classifier extends Serializable{
        def getTermWeight(tf: Int, df : Int, doc_count: Long) : Double
        def fit(data : RDD[LabeledPoint]) 
        def predict(p: Vector) : Int
    }

    class NaiveBayesClassifier extends Classifier{
        var _model : NaiveBayesModel = null
        def getTermWeight(tf: Int, docf : Int, doc_count: Long) = {
            tf.toDouble
        }

        def fit(data : RDD[LabeledPoint]) {
            _model = NaiveBayes.train(data, 1.0)
        }

        def predict(p: Vector) : Int = {
            Math.round(_model.predict(p)).toInt
        }
    }

    class DecisionTreeClassifier(classes : Int) extends Classifier{
        val _impurity = "gini"
        val _maxDepth = 20
        val _maxBins = 16
        val _classes = classes
        var _model : DecisionTreeModel = null

        def getTermWeight(tf: Int, docf : Int, doc_count: Long) =  {
            tf * Math.log((doc_count.toDouble + 1)/ (docf + 1))
        }

        def fit(data : RDD[LabeledPoint]) {
            _model = DecisionTree.trainClassifier(
                    data, 
                    _classes,
                    Map[Int, Int](), //Empty indicates all features are continuous.
                    _impurity, _maxDepth, _maxBins)
        }
        def predict(p: Vector) : Int = {
            math.round(_model.predict(p)).toInt
        }        
    }

    class RandomForestClassifier(classes : Int) extends Classifier{
        val _impurity = "gini"
        val _maxDepth = 20
        val _maxBins = 32
        val _classes = classes
        val _numTrees = 100
        val _seed = 34567; 

        var _model : RandomForestModel = null

        def getTermWeight(tf: Int, docf : Int, doc_count: Long) =  {
            tf * Math.log((doc_count.toDouble + 1)/ (docf + 1))
        }

        def fit(data : RDD[LabeledPoint]) {
            _model = RandomForest.trainClassifier(
                    data, 
                    _classes,
                    Map[Int, Int](), //Empty indicates all features are continuous.
                    _numTrees, 
                    // Let the algorithm choose, which set of features to be made as subsets
                    "auto",
                    _impurity, _maxDepth, _maxBins, _seed)
        }
        def predict(p: Vector) : Int = {
            math.round(_model.predict(p)).toInt
        }        
    }
    def main(args: Array[String]) {
        //random-forest, decision-tree, naive-bayes
        new TextClassifier().classify("naive-bayes")
    }
}

