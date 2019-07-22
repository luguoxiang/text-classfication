import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from numpy.random import seed
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

loaded=np.load('data_tv.npz')

data = loaded['data']
print("Sparsity: {}".format(np.sum(data>0)/(data.shape[0]* data.shape[1])))

train_X, test_X, train_y, test_y = train_test_split(data, loaded['target'], test_size=0.33, random_state=42)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

embedding_dim = 40

CATEGORY_NUM = np.max(train_y) + 1
DOC_MAX_LEN = train_X.shape[1]
VOCABULARY_SIZE = int(np.max([np.max(train_X), np.max(test_X)])) + 1

model = Sequential()

model.add(layers.Embedding(input_dim=VOCABULARY_SIZE, 
                           output_dim=embedding_dim, 
                           input_length=DOC_MAX_LEN))
model.add(layers.Conv1D(128 , 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(CATEGORY_NUM, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


history = model.fit(train_X, train_y,
                    epochs=10,
                    verbose=True,
                    validation_data=(test_X, test_y),
                    batch_size=1000)
loss, accuracy = model.evaluate(train_X, train_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_X, test_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
