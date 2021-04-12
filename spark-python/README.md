# text-classfication using Spark ML and AWS EMR

1. Upload your documents to Amazon S3:
```
git clone http://github.com/luguoxiang/text-classfication.git
python spark-python/upload.py <documents root directory>
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
sudo yum update -y
sudo yum install git
git clone http://github.com/luguoxiang/text-classfication.git
spark-submit --deploy-mode client --master yarn text-classfication/spark-python/spark_text_cls.py
```
The command will train and validate a naive bayes classifier and print accurary:
```
model accuracy 0.874113977884888
```
