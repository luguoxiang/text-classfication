# text-classfication

Put documents which need to be classfied under a directory, which each sub directory contains all documents belonging to certain category. For example: https://www.sogou.com/labs/resource/cs.php
```
pip install -r requirements.txt
python prepare_tf.py <documents root directory>
python text_cls.py <classify method>
```

You can tunning the parameter in text_cls.py:

```
classifier_map = {
   "naive-bayes": cls_naive_bayes(),
   "linear-svm": cls_linear_svm(C=1),
   "random-forest": cls_random_forest(max_term=5000, n_estimators=100),
   "neural-network": cls_neural_network(learning_rate=0.01, n_epochs = 20, batch_size = 1000),
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
