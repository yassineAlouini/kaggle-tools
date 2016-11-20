import xgboost
from sklearn.cross_validation import train_test_split
from xgboost import XGBRegressor

# Some constants

TEST_SIZE = 0.2

# Create a linear gradient boosting regressor model with XGBoost


def simple_linear_model(X, y, hyperparameters):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE)
    lm = XGBRegressor(**hyperparameters)
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    return predictions
