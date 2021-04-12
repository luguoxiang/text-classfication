# text-classfication using Spark ML and Hadoop sparn cluster
Put documents which need to be classfied under a directory of Hadoop HDFS, which each sub directory contains all documents belonging to certain category(The documents should use utf-8 encoding.)

```
mvn package
spark-submit --class com.guoxiang.hdfs.TextClassifier --master yarn --deploy-mode client target/test-hdfs-jar-with-dependencies.jar
```

The command will train and validate a naive bayes classifier and print accurary:
```
model accuracy 0.874113977884888
```

You can change to decision tree or random forest by changing following code in main:
```
        //random-forest, decision-tree, naive-bayes
        new TextClassifier().classify("naive-bayes")
```

