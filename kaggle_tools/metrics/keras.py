import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import fbeta_score

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
DEFAULT_THRESHOLDS = np.linspace(START, END, N_STEPS)
DEFAULT_BETA = 1
DEFAULT_LOGS = {}
FBETA_METRIC_NAME = "val_fbeta"

# Notice that this callback only works with Keras 2.0.0


class FBetaMetricCallback(Callback):

    def __init__(self, beta=DEFAULT_BETA, thresholds=DEFAULT_THRESHOLDS):
        self.beta = beta
        self.thresholds = thresholds
        # Will be initialized when the training starts
        self.val_fbeta = None

    def on_train_begin(self, logs=DEFAULT_LOGS):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_fbeta = []

    def _score_per_threshold(self, predictions, targets, threshold):
        """ Compute the Fbeta score per threshold.
        """
        # Notice that here I am using the sklearn fbeta_score function.
        # You can read more about it here:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        thresholded_predictions = (predictions > threshold).astype(int)
        return fbeta_score(targets, thresholded_predictions, beta=self.beta)

    def on_epoch_end(self, epoch, logs=DEFAULT_LOGS):
        val_predictions = self.model.predict(self.validation_data[0])
        val_targets = self.validation_data[1]
        _val_fbeta = np.mean([self._score_per_threshold(val_predictions,
                                                        val_targets, threshold)
                              for threshold in self.thresholds])
        self.val_fbeta.append(_val_fbeta)
        print("Current F{} metric is: {}".format(str(self.beta), str(_val_fbeta)))
        return

    def on_train_end(self, logs=DEFAULT_LOGS):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history[FBETA_METRIC_NAME] = self.val_fbeta
