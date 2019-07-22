import scipy.sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import sys
from tensorflow import keras

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

class TensorflowOneLayerNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs, batch_size):
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        n_inputs = X.shape[1]
        n_outputs = np.max(y) + 1

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(n_outputs, input_shape=(n_inputs,), 
		activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
        model.summary()
        model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size)

        model_json = model.to_json()
        with open("tf_nn.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("tf_nn.h5")
        return self

    def predict(self, X):
        json_file = open('tf_nn.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights("tf_nn.h5")

        model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

        return model.predict_classes(X)

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
        ('term_compressor', VocabularyCompressorByTF(max_term)), 
        ('tf-idf', TfIdfConvertor()),
        ('svc', RandomForestClassifier(n_estimators=n_estimators,  random_state=0, verbose=True)),
   ])

def cls_neural_network(n_epochs, batch_size):
    return Pipeline([
        ('idf_compressor', VocabularyCompressorByIDF()),
        ('tf-idf', TfIdfConvertor()),
         ('normalizer', Normalizer()), 
        ("nn", TensorflowOneLayerNN(n_epochs, batch_size)),
    ])

classifier_map = {
   "naive-bayes": cls_naive_bayes(),
   "linear-svm": cls_linear_svm(C=1),
   "random-forest": cls_random_forest(max_term=5000, n_estimators=100),
   "neural-network": cls_neural_network(n_epochs = 20, batch_size=100),
}

if len(sys.argv) < 2 or sys.argv[1] not in classifier_map:
    print("python text_cls.py [{}]".format(' | '.join(classifier_map)))
    sys.exit(1)

classifier=classifier_map[sys.argv[1]]

XX = scipy.sparse.load_npz('data_tf.npz')
yy = np.load('target.npz')['target']
print("data shape: {}, target shape: {}".format(XX.shape, yy.shape))

score = cross_val_score(classifier, XX, yy, cv=3, scoring="accuracy")

#train_X, test_X, train_y, test_y = train_test_split(XX, yy, test_size=0.33, random_state=42)
#classifier.fit(train_X, train_y)
#predict_y= classifier.predict(test_X)
#score = accuracy_score(test_y,predict_y)

print("Accuracy score: {}".format(score))
