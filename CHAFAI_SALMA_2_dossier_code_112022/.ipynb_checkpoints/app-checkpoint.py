# 1. Library imports
# Import Needed Libraries
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
import pickle
import dill
from fastapi import FastAPI

# 2. Create the app object
app = FastAPI()

# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
pickle_in = open("model_credit_fr.pkl","rb")
classifier=pickle.load(pickle_in)
features_selected=pickle.load(open("features_selected.pkl","rb"))


# Téléchargement de shap explainer
#explainer = pickle.load(open("shap_explainer.pkl","rb"))
shap_explainer = dill.load(open("shap_explainer_fr.dill","rb"))

# Chargement d'un échantillon de 100 clients
df = pd.read_csv('echantillon.csv')

# Récupérer la liste des colonnes du jeu de données
columns = features_selected




# Récupérer la liste des id_clients
l_id_client = df['SK_ID_CURR'].tolist()

# Définir une route qui retourne la liste des id clients
@app.get("/clients")
def ids_route():
    return {"id_clients": l_id_client}

@app.get("/client/{id_client}")
def get_client(id_client: int):
    # récupérer les données de la base de données en utilisant l'id_client
    X = df[df["SK_ID_CURR"] == id_client]
    
    
    # vérifier si les données ont été trouvées
    if X.empty:
        raise HTTPException(status_code=404, detail=f"Client with id_client {id_client} not found")
    
    # renvoyer les données sous forme de dictionnaire
    return X.to_dict(orient="records")

# Définir une route qui retourne la liste des colonnes
@app.get("/columns")
def columns_route():
    return {"columns": columns}

@app.get("/column/{column_name}")
def column_route(column_name: str):


    # Récupérer la valeur de la colonne
    column_values = df[column_name].values.tolist()

    # La target
    y = df["TARGET"].values.tolist()
    # liste de chaînes de caractères correspondantes
    labels = ["solvable","non_solvable"]
    
    y = list(map(lambda x: labels[x], y))
    return {"column_name": column_name, "column_values": column_values, "target" : y}



# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.post('/predict')
def predict(id_client: int):
    # On récupère les informations du client
    ligne = df[df['SK_ID_CURR'] == id_client]
    X_test = ligne[features_selected]
    
    prediction = classifier.predict(X_test)
    pred_proba = classifier.predict_proba(X_test)
    
    # Map prediction to appropriate label
    prediction_label = ["Crédit accordé" if prediction == 0 else "Crédit refusé"]
    # Return response back to client
    return {"prediction": prediction_label[0],
           "probabilité" : max(pred_proba[0])}


# A revoir
#@app.post('/test_data')
#def test_data(id_client: int):
    # Récupérez les données d'entrée pour le client sélectionné
#    ligne = df[df['SK_ID_CURR'] == id_client]
#    X_test = ligne[features_selected]
#    return X_test

@app.get("/data")
def get_data():
    # sélectionner les colonnes spécifiées du DataFrame
    cols = features_selected
    df_selected = df[cols]
    
    # convertir le DataFrame en dictionnaire
    data = df_selected.to_dict(orient="records")
    
    return {"data": data}


@app.post("/predict_explainer/{id_client}")
def predict_explain(id_client: int) :
    # Récupérer les données d'entrée pour le client sélectionné
    #ligne = df[df["SK_ID_CURR"] == id_client]
    X_test = df[features_selected]
    X_test = X_test.to_dict(orient="records")

    # Calculer les valeurs SHAP pour les prédictions de test
    shap_values = shap_explainer.shap_values(X_test)

    # Construire un dictionnaire contenant les explications de modèle
    return {"X_test": X_test.tolist(), "shap_values": shap_values}

















#@app.get("/explainer")
#def get_explainer():
#    explainer = shap.TreeExplainer(classifier)
#    explainer_dict = explainer.to_dict()
#    return explainer_dict

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)