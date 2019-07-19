import scipy.sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import sys
import tensorflow as tf

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
    def __init__(self, learning_rate, n_epochs, batch_size):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def to_sparse_tf_value(self, value):
        value = value.tocoo()
        return  tf.SparseTensorValue(
            indices=np.array([value.row, value.col]).T,
            values=value.data,
            dense_shape=value.shape)

    def build_nn(self, X, y, n_inputs, n_outputs):
        stddev = 2 / np.sqrt(n_inputs + n_outputs)
        init = tf.truncated_normal((n_inputs, n_outputs), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_outputs]), name="bias")
        logits = tf.sparse.sparse_dense_matmul(X, W) + b

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
    
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        return logits, loss, training_op, accuracy

    def fit(self, X, y):
        tf.reset_default_graph()
        n_inputs = X.shape[1]
        n_outputs = np.max(y) + 1

        X_holder = tf.sparse_placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y_holder = tf.placeholder(tf.int64, shape=(None), name="y")

        logits, loss, training_op, accuracy = self.build_nn(X_holder, y_holder, n_inputs, n_outputs)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for iteration in range(n_inputs // self.batch_size):
                    X_batch = X[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    y_batch = y[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    sess.run(training_op, feed_dict={X_holder: self.to_sparse_tf_value(X_batch), y_holder: y_batch})

                acc_train = accuracy.eval(feed_dict={X_holder: self.to_sparse_tf_value(X), y_holder: y})
                print(epoch, "Train accuracy:", acc_train)

            save_path = saver.save(sess, "./model_final.ckpt")
        return self

    def predict(self, X):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph("./model_final.ckpt.meta")
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            saver.restore(sess, "./model_final.ckpt")
            logits=graph.get_tensor_by_name('SparseTensorDenseMatMul/SparseTensorDenseMatMul:0')
            X_indices = graph.get_tensor_by_name("X/indices:0")
            X_values = graph.get_tensor_by_name("X/values:0")
            X_shape = graph.get_tensor_by_name("X/shape:0")
            value = X.tocoo()
            result = logits.eval(feed_dict={
		X_indices: np.array([value.row, value.col]).T,
		X_values: value.data,
		X_shape: value.shape})
            return np.argmax(result, axis=1)

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

def cls_neural_network(learning_rate, n_epochs, batch_size):
    return Pipeline([
        ('idf_compressor', VocabularyCompressorByIDF()),
        ('tf-idf', TfIdfConvertor()),
         ('normalizer', Normalizer()), 
        ("nn", TensorflowOneLayerNN(learning_rate, n_epochs, batch_size)),
    ])

classifier_map = {
   "naive-bayes": cls_naive_bayes(),
   "linear-svm": cls_linear_svm(C=1),
   "random-forest": cls_random_forest(max_term=5000, n_estimators=100),
   "neural-network": cls_neural_network(learning_rate=0.01, n_epochs = 20, batch_size = 1000),
}

if len(sys.argv) < 2 or sys.argv[1] not in classifier_map:
    print("python text_cls.py [{}]".format(' | '.join(classifier_map)))
    sys.exit(1)

classifier=classifier_map[sys.argv[1]]

XX = scipy.sparse.load_npz('data_tf.npz')
yy = np.load('target.npz')['target']
print("data shape: {}, target shape: {}".format(XX.shape, yy.shape))

score = cross_val_score(classifier, XX, yy, cv=3, scoring="accuracy")

#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
#classifier.fit(train_X, train_y)
#predict_y= classifier.predict(test_X)
#score = accuracy_score(test_y,predict_y)

print("Accuracy score: {}".format(score))
