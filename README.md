# text-classfication

Put documents which need to be classfied under a directory, which each sub directory contains all documents belonging to certain category. For example: https://www.sogou.com/labs/resource/cs.php
```
python prepare_tf.py <documents root directory>
python text_cls.py <classify method>
```
| classify method | description |
| --------------- | ----------- |
| naive-bayes | Multinomial naive bayes classifier |
| linear-svm | Linear SVM classifier |
| random-forest | random forest classifier |
| neural-network | single layer NN using tensorflow |

The script will do three fold cross validation on the documents and print accuracy.
