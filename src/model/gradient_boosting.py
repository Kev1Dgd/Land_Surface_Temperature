from sklearn.ensemble import GradientBoostingRegressor

def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
