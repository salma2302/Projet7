import streamlit as st
import json
import requests
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
import numpy as np
import pickle
import altair as alt
import dill

#Application hello word
st.title("Prédiction de l'accord d'un crédit d'un client")
#st.subheader("Données des cliens")
st.markdown(":tada: Cette app affiche si ou non on accorde un prêt à un client")

st.image("photo_entreprise.png", width=200)




# Chargement d'un échantillon de 100 clients
#df = pd.read_csv('echantillon.csv')
#df.drop(columns = 'Unnamed: 0', inplace=True)

# Téléchargement de shap explainer
#explainer = pickle.load(open("shap_explainer.pkl","rb"))
shap_explainer = dill.load(open("shap_explainer_fr.dill","rb"))

# Chargement du jeu de données servi pout l'entrainement (finir cette partie)
#train = pd.read_csv('df_train.csv')

# Choix de l'identifiant
st.sidebar.header("Choix du client")
res_id = requests.get("http://localhost:8000/clients")
id_client = res_id.json()["id_clients"]
id_selected = st.sidebar.selectbox("Choix de l'id du client", options=id_client)

# La liste déroulante des features
#option_features = st.sidebar.selectbox('Sélection ta variable 1',tuple(liste_cols))
# Appeler la route qui retourne la liste des colonnes
res_col = requests.get("http://localhost:8000/columns")
columns = res_col.json()["columns"]


#ligne = df[df['SK_ID_CURR'] == id_selected]

# On crée la première liste déroulante
columns_selected1 = st.sidebar.selectbox('Sélectionnez la première variable :', options=columns)
columns_disponibles = columns.copy()

# Enlever la variable sélectionnée de la liste des variables disponibles
columns_disponibles.remove(columns_selected1)

# Utiliser la liste des variables disponibles pour afficher la deuxième liste déroulante
columns_selected2 = st.sidebar.selectbox('Sélectionnez la deuxième variable :', options=columns_disponibles)

# Afficher les variables sélectionnées
st.write('Vous avez sélectionné les variables suivantes :', columns_selected1, 'et', columns_selected2)


# L'inputs 
inputs = {"id_client" : id_selected}

#---------------------------------------------
## Les infos du clients 
response_client = requests.get(f"http://localhost:8000/client/{id_selected}")

# vérifier si la réponse est valide
if response_client.status_code == 200:
    # récupérer les données du client sous forme de dictionnaire
    client_data = response_client.json()[0]
    
    client_data = pd.DataFrame.from_dict(client_data, orient='index').transpose()
    #st.write(client_data)
else:
    st.error("Error getting client data")
    
#---------------------------------------------



# Les colonnes qui ont servi pour l'entraînement
#liste_cols = features_selected

# Téléchargement de shap explainer
#explainer = pickle.load(open("shap_explainer_fr.pkl","rb"))

#liste_cols = list(train.columns)


# On récupère les informations du client
#ligne = df[df['SK_ID_CURR'] == option]
#X_test = ligne[liste_cols]

st.subheader('Voici les infos du client')
st.table(client_data)

if st.sidebar.button("Graphique univarié") :
    
    response1 = requests.get(f"http://localhost:8000/column/{columns_selected1}")
    response2 = requests.get(f"http://localhost:8000/column/{columns_selected2}")
    
    if response1.status_code == 200 and response2.status_code == 200:
        # Récupérer les données de la réponse
        df1 = response1.json()
        df2 = response2.json()
        
        point = client_data[[columns_selected1, columns_selected2]]
        
        # Créez les deux graphes avec Plotly
        fig1 = px.box(df1, x='column_values')
        fig2 = px.box(df2, x='column_values')
        
        # Créez un trace avec les données du client
        trace1 = go.Scatter(x=point[columns_selected1], y=[0], mode='markers', name='Client', marker=dict(color='#e7298a', size=10))
        trace2 = go.Scatter(x=point[columns_selected2], y=[0], mode='markers', name='Client', marker=dict(color='#e7298a', size=10))
        
        # Ajoutez le trace au figure existant
        fig1.add_trace(trace1)
        fig2.add_trace(trace2)
        
        fig1.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected1}",
            xaxis_title= f"La variable {columns_selected1}",
            legend_title="Legend Title"
)
        
        
        fig2.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected2}",
            xaxis_title= f"La variable {columns_selected2}",
            legend_title="Legend Title"
)





        # Affichez les figures côte à côte
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        
    
    



if st.sidebar.button("Graphique bivariée") :
    
    response1 = requests.get(f"http://localhost:8000/column/{columns_selected1}")
    response2 = requests.get(f"http://localhost:8000/column/{columns_selected2}")

    # Afficher les colonnes dans l'interface utilisateur
    # Vérifier le code de retour de la réponse
    if response1.status_code == 200 and response2.status_code == 200:
        # Récupérer les données de la réponse
        data1 = response1.json()
        data2 = response2.json()
        
        point = client_data[[columns_selected1, columns_selected2]]
        
        # Afficher les données dans l'interface utilisateur
        #st.write(f"Colonne : {data['column_name']} et sa valeur : {data['column_values']}")
        
        # Créer un graphique à barres en utilisant plotly
        fig = px.scatter(x=data1['column_values'], y=data2['column_values'],
                         color=data1["target"],
                         color_discrete_sequence=["green", "red"]
        )
        fig.add_scatter(x=point[columns_selected1], y=point[columns_selected2],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                       name='Client')
        
        fig.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected1} en fonction de la variable {columns_selected2}",
            xaxis_title= f"La variable {columns_selected1}",
            yaxis_title= f"La variable {columns_selected2}",
            legend_title="Legend Title"
)
        
        # Afficher le graphique dans l'interface utilisateur
        st.plotly_chart(fig, use_container_width=True)
        



    
    #res2 = requests.get(f"http://localhost:8000/column/{columns_selected}")
    #st.write(res2.text())

#@st.cache

st.sidebar.header("Prédiction")

if st.sidebar.button("predict") :
    res = requests.post(url = "http://127.0.0.1:8000/predict", params=inputs)
    prediction = res.json()
    
    
    # Affichage du résultat à l'utilisateur
    #st.subheader("Le résultat de la prédiction: ")
    #st.success(f"The prediction from model: {prediction['prediction']} avec une probabilité de {prediction['probabilité']}")
    st.success(f"Le crédit est {prediction['prediction']} avec une proba de {prediction['probabilité']} pour le client avec l'id {id_selected}")
    
       
    # Affichage du résultat à l'utilisateur
    #st.success(f"Prediction: {result['prediction']}")  
    




        
        
# A revoir        
if st.sidebar.button("Features importance") :

    #response2 = requests.post('http://127.0.0.1:8000/predict_explain', json={"X_test": X_test})
    #response = requests.post(f"http://127.0.0.1:8000/predict_explainer/{id_selected}")
    response = requests.get("http://localhost:8000/data")
    if response.status_code == 200:
        #explainer_dict = response2.json()
        X_test = response.json()["data"] 
        X_test = pd.DataFrame(X_test)
        

        shap_values = shap_explainer.shap_values(X_test)

        st.subheader("Valeurs SHAP")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist()))
        
        
        shap_val = shap_explainer.shap_values(client_data)
        # Créer un trace Plotly avec les données de l'explication SHAP

        # Afficher le trace avec st.plotly_chart
        shap.plots._waterfall.waterfall_legacy(shap_explainer.expected_value[0],
                                               shap_val[0][0],
                                               feature_names = client_data.columns,
                                                max_display= 10) 
        st.pyplot()
        

        
    else :
        st.write("Erreur: la requête a échoué avec le code d'état", response.status_code)
    



         
         
