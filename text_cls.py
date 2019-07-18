import scipy.sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sys

class VocabularyCompressorByIDF(BaseEstimator, TransformerMixin):
    def __init__(self, min_idf = 5, max_idf_rate = 0.4):
        self.min_idf = min_idf
        self.max_idf_rate = max_idf_rate    

    def fit(self, X, y=None):
        D=X.shape[0]

        #remove term which idf<min_idf and idf >max_idf_rate * D
        idf=np.asarray((X!=0).sum(axis=0))[0]
        self.filter = (idf >=self.min_idf) & (idf < self.max_idf_rate * D)
        return self

    def transform(self, X, y=None):
        result = X.tocsc()[:,self.filter]
        print("VocabularyCompressorByIDF: {}".format(result.shape[1]))
        return result


class TfIdfConvertor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        idf = X.shape[0] / ((X != 0).sum(axis=0) + 0.01)
        return X.multiply(np.log(idf))

class VocabularyCompressorByTF(BaseEstimator, TransformerMixin):
    def __init__(self, max_term):
        self.max_term = max_term
    
    def fit(self, X, y=None):
        weight=np.asarray(X.sum(axis=0))[0]
        threthold = sorted(weight, reverse=True)[self.max_term]
        self.filter = (weight >= threthold)
        return self 

    def transform(self, X, y=None):
        result = X.tocsc()[:,self.filter]
        print("VocabularyCompressorByTF: {}".format(result.shape[1]))
        return result

def cls_naive_bayes():
    from sklearn.naive_bayes import MultinomialNB
    return Pipeline([
        ('idf_compressor', VocabularyCompressorByIDF()),
        ('nbc',  MultinomialNB()),
    ])

def cls_linear_svm(C):
    from sklearn.svm import LinearSVC

    return Pipeline([
        ('idf_compressor', VocabularyCompressorByIDF()),
        ('tf-idf', TfIdfConvertor()),
        ('normalizer', Normalizer()), 
        ('svc', LinearSVC(C=C,loss="hinge", max_iter=5000, verbose=True)),
    ])

def cls_random_forest(max_term, n_estimators):
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([
        ('idf_compressor', VocabularyCompressorByIDF()),
        ('tf-idf', TfIdfConvertor()),
        ('term_compressor', VocabularyCompressorByTF(max_term)), 
        ('svc', RandomForestClassifier(n_estimators=n_estimators,  random_state=0, verbose=True)),
   ])


classifier_map = {
   "naive-bayes": cls_naive_bayes(),
   "linear-svm": cls_linear_svm(C=1),
   "random-forest": cls_random_forest(max_term=5000, n_estimators=100),
}

if len(sys.argv) < 2 or sys.argv[1] not in classifier_map:
    print("python text_cls.py [{}]".format(' | '.join(classifier_map)))
    sys.exit(1)

classifier=classifier_map[sys.argv[1]]

X = scipy.sparse.load_npz('data_tf.npz')
y = np.load('target.npz')['target']
print("data shape: {}, target shape: {}".format(X.shape, y.shape))

score = cross_val_score(classifier, X, y, cv=3, scoring="accuracy")
print("Accuracy score: {}".format(score))
