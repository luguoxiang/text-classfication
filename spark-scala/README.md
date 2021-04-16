# text-classfication using Spark ML and Hadoop yarn cluster

Put documents which need to be classfied under a directory of Hadoop HDFS, which each sub directory contains all documents belonging to certain category(The documents should use utf-8 encoding.)

```
mvn package
export YARN_CONF_DIR=...
export HADOOP_CONF_DIR=...
spark-submit --class com.guoxiang.hdfs.TextClassifier --master yarn --deploy-mode client target/test-hdfs-jar-with-dependencies.jar
```

The command will train and validate a svm classifier and print accurary:
```
Test set accuracy = 0.8976245210727969
```

You can change to naive bayes or random forest by changing following code in main:
```
        //randomForestClassifier, naiveBayes
        linearSVC.fit(training).transform(test),
```

for naiveBayes, you also need to use TfTermWeight:
```
        var termWeight = new TfTermWeight()
        //var termWeight = new TfIdfTermWeight()
```
Also, change following const as your needs:
```
    val TEXT_BASE_DIR = "/news_text"
    val HDFS_URL = "hdfs://hadoop-master:9000"
    val HDFS_USER = "hduser"
    val TEXT_ENCODING = "gb2312"
```
