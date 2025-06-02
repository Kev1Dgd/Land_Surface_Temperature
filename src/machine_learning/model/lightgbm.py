from lightgbm import LGBMRegressor

def train_lightgbm(X_train, y_train):
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model