# text-classfication

Put documents which need to be classfied under a directory, which each sub directory contains all documents belonging to certain category. For example: https://www.sogou.com/labs/resource/cs.php
```
pip install -r requirements.txt
python prepare_tf.py <documents root directory>
python text_cls.py <classify method>
```

You can tune the parameter in text_cls.py:

```
classifier_map = {
   "naive-bayes": cls_naive_bayes(),
   "linear-svm": cls_linear_svm(C=1),
   "random-forest": cls_random_forest(max_term=5000, n_estimators=100),
   "neural-network": cls_neural_network(n_epochs = 20, batch_size = 1000),
}
```

| classify method | description | library |
| --------------- | ----------- | ------- |
| naive-bayes | Multinomial naive bayes classifier | sklearn.naive_bayes.MultinomialNB |
| linear-svm | Linear SVM classifier | sklearn.svm.LinearSVC |
| random-forest | random forest classifier | sklearn.ensemble.RandomForestClassifier |
| neural-network | single layer neural network | tensorflow |

The script will do three fold cross validation on the documents and print accuracy.

# Deep learning text-classfication

Word Embedding + CNN + 2 Dense layer

```
python prepare_tv.py <documents root directory>
python text_cls_deep.py 
```

# text-classfication using Spark ML and AWS EMR

1. Upload your documents to Amazon S3:
```
git clone http://github.com/luguoxiang/text-classfication.git
python spark/upload.py <documents root directory>
```

2. Create following script file and upload it to Amazon S3
```#!/bin/bash
sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
sudo easy_install-3.6 pip
sudo /usr/local/bin/pip3 install jieba nltk boto3
```
 
3. Create or config Spark cluster in AmazonEMR:

* Create ssh key named spark-key for spark cluster(https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair)
* Create spark cluster:
```
aws emr create-cluster --name "spark-cluster" --release-label emr-5.28.0 --applications Name=Spark --ec2-attributes KeyName=spark-key --instance-type m5.xlarge --instance-count 3 --use-default-roles  --bootstrap-actions Path="s3://...path to uploaded script file..."
```

4. Login to spark master node and run
```
git clone http://github.com/luguoxiang/text-classfication.git
spark-submit --deploy-mode client --master yarn text-classfication/spark/spark_text_cls.py
```
The command will train and validate a naive bayes classifier and print accurary:
```
model accuracy 0.874113977884888
```
