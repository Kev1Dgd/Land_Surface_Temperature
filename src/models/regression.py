# src/models/regression.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def perform_regression(fluxnet_data, amsre_data):
    """Effectue une régression linéaire pour prédire la température de surface à partir de la température de brillance."""
    
    # Filtrage des données valides
    valid_fluxnet = fluxnet_data.dropna(subset=['surface_temperature'])
    valid_amsre = amsre_data.dropna(subset=['brightness_temp_37v'])

    # Associer les données Fluxnet et AMSR-E sur la latitude et longitude
    merged_data = valid_fluxnet.merge(valid_amsre, on=['latitude', 'longitude'], how='inner')
    
    # Sélectionner les variables indépendantes (température de brillance) et dépendantes (température de surface)
    X = merged_data['brightness_temp_37v'].values.reshape(-1, 1)  # Variables indépendantes
    y = merged_data['surface_temperature'].values  # Variable dépendante (température de surface)
    
    # Création du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    # Affichage des coefficients de la régression
    print(f"Coefficients de régression : {model.coef_}")
    print(f"Ordonnée à l'origine : {model.intercept_}")
    
    # Prédictions
    predictions = model.predict(X)
    
    # Affichage de la régression
    plt.scatter(X, y, color='blue', label='Données réelles')
    plt.plot(X, predictions, color='red', label='Régression')
    plt.xlabel("Température de brillance (K)")
    plt.ylabel("Température de surface (°C)")
    plt.title("Régression linéaire : Température de brillance vs Température de surface")
    plt.legend()
    plt.show()

    return model
