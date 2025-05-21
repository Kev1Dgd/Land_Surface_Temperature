from sklearn.svm import SVR

def train_svr(X_train, y_train, kernel="rbf", C=1.0, epsilon=0.2):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train.ravel())
    return model
