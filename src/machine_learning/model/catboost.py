from catboost import CatBoostRegressor

def train_catboost(X_train, y_train):
    model = CatBoostRegressor(verbose=0, iterations=100, learning_rate=0.1, depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model