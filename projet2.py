from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pandas as pd
import streamlit as st
import pickle  


def Charger_Datas():
    st.subheader("📁 Fournissez le fichier CSV contenant les données d’analyse 📊📁")
    fichier = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if fichier is not None:
        with st.spinner("Chargement de nos datas 🙃..."):
            try:
                df = pd.read_csv(fichier)
                st.success("✅ Fichier chargé avec succès !")
                st.write("# Visualisation des données :")
                st.dataframe(df)
                return df
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")
                return None
    else:
        st.info("💡 Veuillez fournir les données dans un fichier CSV pour continuer.")
        return None

def travail_classification(df):
    st.subheader("🎯 Travail de Classification")
    colonnes = df.columns.tolist()
    cible = st.selectbox(" Quelle est la variable que vous ciblez ", colonnes)

    if cible:
        X = df.drop(columns=[cible])
        Y = df[cible]

        test_size = 0.3
        seed = 42
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        models = {
            "Régression Logistique": LogisticRegression(solver='newton-cg'),
            "K Plus Proches Voisins C": KNeighborsClassifier(),
            "Arbre de Décision": DecisionTreeClassifier(),
            "Analyse Discriminante Linéaire": LinearDiscriminantAnalysis()
        }

        st.write("### 📊 Résultats des modèles")
        results = []

        with st.spinner("loading classification training 🙃..."):
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                results.append({
                    "Modèle": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 2),
                    "Précision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 2),
                    "Rappel": round(recall_score(y_test, y_pred, average='weighted'), 2),
                    "F1-Score": round(f1_score(y_test, y_pred, average='weighted'), 2)
                })
                sauvegarder_modele(model, name)
                
        st.dataframe(pd.DataFrame(results))

        if "predire_clicked" not in st.session_state:
            st.session_state.predire_clicked = False

        if st.button("Cliquez pour ouvrir l'Interface de Prédiction"):
            st.session_state.predire_clicked = True

        if st.session_state.predire_clicked:
            travail_prediction_C(df, cible)

def travail_regression(df):
    st.subheader("🎯 Travail de Régression")
    colonnes = df.columns.tolist()
    cible = st.selectbox(" Quelle est la variable que vous ciblez ", colonnes)

    if cible:
        X = df.drop(columns=[cible])
        Y = df[cible]

        test_size = 0.3
        seed = 42
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        models = {
            "Régression Linéaire": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K Plus Proches Voisins R": KNeighborsRegressor()
        }

        st.write("### 📊 Résultats des modèles de régression")
        results = []

        with st.spinner("loading regression training 🙃..."):
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                mae = round(mean_absolute_error(y_test, y_pred), 3)
                mse = round(mean_squared_error(y_test, y_pred), 3)
                r2 = round(r2_score(y_test, y_pred), 3)

                results.append({
                    "Modèle": name,
                    "MAE": mae,
                    "MSE": mse,
                    "R²": r2
                })

                sauvegarder_modele(model, name)

        st.dataframe(pd.DataFrame(results))

        if "predire_clicked" not in st.session_state:
            st.session_state.predire_clicked = False

        if st.button("Cliquez pour ouvrir l'Interface de Prédiction"):
            st.session_state.predire_clicked = True

        if st.session_state.predire_clicked:
            travail_prediction_R(df, cible)

def travail_prediction_C(df, val_pred):
    st.subheader("🔮 Interface de Prédiction 🔮")
    st.write("### Entrez les données pour la prédiction")
    modeles_dispo = st.selectbox("Sélectionnez un modèle déjà entraîné", ["Régression Logistique", "K Plus Proches Voisins C", "Arbre de Décision", "Analyse Discriminante Linéaire"])

    colonnes = df.columns.tolist()
    valeurs_entrees = {}
    for col in colonnes:
        if col != val_pred : 
            valeurs_entrees[col] = st.number_input(f"Entrez la valeur pour {col}:", value=0)

    if st.button("Prédire"):
        modele = charger_modele(f"{modeles_dispo}_model.pkl")
    
        if modele:
            nouveau_df = pd.DataFrame([valeurs_entrees])
            y_pred = modele.predict(nouveau_df)
            st.success(f"🎯 Résultat de la prédiction : {y_pred[0]}")
            X_test = df.drop(columns=[val_pred])
            Y_test = df[val_pred]

            y_pred_all = modele.predict(X_test)

            comparison_results = []

            for instance, prediction, true_value in zip(X_test.values, y_pred_all, Y_test):
                comparison_results.append({
                    "Instance": list(instance),
                    "Prédiction": prediction,
                    "Réel": true_value
                })

            st.write("### 🔍 Comparaison des prédictions sur l’ensemble des données")
            st.dataframe(pd.DataFrame(comparison_results))

def travail_prediction_R(df, val_pred):
    st.subheader("🔮 Interface de Prédiction 🔮")
    st.write("### Entrez les données pour la prédiction")
    modeles_dispo = st.selectbox("Sélectionnez un modèle déjà entraîné",["Régression Linéaire", "Lasso", "Ridge", "K Plus Proches Voisins R"])

    colonnes = df.columns.tolist()
    valeurs_entrees = {}
    for col in colonnes:
        if col != val_pred : 
            valeurs_entrees[col] = st.number_input(f"Entrez la valeur pour {col}:", value=0)
    
    for col in colonnes:
        if col != val_pred:
            valeurs_entrees[col] = st.number_input(f"Entrez la valeur pour {col}:", value=0.0)

    if st.button("Prédire"):
        modele = charger_modele(f"{modeles_dispo}_model.pkl")
    
        if modele:
            nouveau_df = pd.DataFrame([valeurs_entrees])
            y_pred = modele.predict(nouveau_df)
            st.success(f"🎯 Résultat de la prédiction : {y_pred[0]}")
            X_test = df.drop(columns=[val_pred])
            Y_test = df[val_pred]

            y_pred_all = modele.predict(X_test)

            comparison_results = []

            for instance, prediction, true_value in zip(X_test.values, y_pred_all, Y_test):
                comparison_results.append({
                    "Instance": list(instance),
                    "Prédiction": prediction,
                    "Réel": true_value
                })

            st.write("### 🔍 Comparaison des prédictions sur l’ensemble des données")
            st.dataframe(pd.DataFrame(comparison_results))

def charger_modele(model_name):
    try:
        return pickle.load(open(model_name, 'rb'))
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

def sauvegarder_modele(modele, model_name):
    with open(f"{model_name}_model.pkl", 'wb') as file:
        pickle.dump(modele, file)    


def Application_Maniche():
    st.set_page_config(page_title="Application Maniche", layout="centered")
    st.title("Laboratoire d'APPRENTISSAGE AUTOMATIQUE ⛓️📱")
    st.markdown("""
    Cette application vous permet de créer des modèles d'apprentissage automatique selon :
    - La classification
    - La régression
    """)

    df = Charger_Datas()

    if df is not None:
        st.sidebar.title("🧭 Station de Navigation")
        choix = st.sidebar.radio("Sélectionner une tâche :", ["Visualisation des données","Travail de Classification", "Travail de Régression"])

        if choix == "Visualisation des données":
         st.markdown("# Appréciez et Consultez les données 📱✍️")   
        elif choix == "Travail de Classification":
            st.markdown("# Classification ")
            travail_classification(df)
        elif choix == "Travail de Régression":
            st.markdown("# Regression ")
            travail_regression(df)

if __name__ == "__main__":
    Application_Maniche()
