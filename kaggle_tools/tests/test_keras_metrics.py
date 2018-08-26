"""
Create a simple Keras model and test that the metrics work as expected.
"""

# Some of these tests are inspired from here:
# https://github.com/keras-team/keras/blob/master/tests/keras/test_callbacks.py

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.test_utils import get_test_data, keras_test

from kaggle_tools.metrics.keras import FBETA_METRIC_NAME, FBetaMetricCallback

input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20
SEED = 42
TEST_BETA = 2
EPOCHS = 5


@keras_test
def test_fbeta_metric_callback():
    np.random.seed(SEED)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_classes)
    # Simple classification model definition
    # TODO: Refactor this into a function.
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    fbeta_metric_callback = FBetaMetricCallback(beta=TEST_BETA)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[fbeta_metric_callback], epochs=EPOCHS)
    assert fbeta_metric_callback.val_fbeta is not None
    assert FBETA_METRIC_NAME in history.history.keys()
    assert history.history[FBETA_METRIC_NAME] is not None
    assert len(history.history[FBETA_METRIC_NAME]) == EPOCHS
