from sklearn.cross_validation import train_test_split

RANDOM_STATE = 42  #  For reproducibility


def split_validation_set(train, target, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=test_size, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test
