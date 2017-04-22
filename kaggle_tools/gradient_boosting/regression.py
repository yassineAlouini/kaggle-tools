import xgboost
from sklearn.cross_validation import train_test_split
from xgboost import XGBRegressor

from kaggle_tools import conf


class GBRegression(object):

    def __init__(self, X, y, conf):
        self.X = X
        self.y = y
        self.conf = conf
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.conf.TEST_SIZE)
        self.ml_model = XGBRegressor()

    def fit(self):
        self.ml_model.fit(self.X_train, self.y_train)

    def predict(self):
        predictions = self.ml_model.predict(X_test)
        return predictions

    def evaluate(self):
        pass

    def cv(self):
        pass
