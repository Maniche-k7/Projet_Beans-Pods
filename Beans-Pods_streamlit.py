import streamlit as st
from datetime import date
from PIL import Image
from pandas import read_csv
import pandas as pd
import scipy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

st.markdown(
        """
    <div style ='text-align:center' >
    <h1 style = 'color: green'><u> Compagnie Beans and Pods </u></h1>
    <h3 style = 'color: blue'> Statistiques descriptives et Analyse de données </h3>
    """, unsafe_allow_html= True
    )

try:
    fichier='BeansDataSet.csv'
    col=['Channel', 'Region' , 'Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']
    data= read_csv(fichier, names=col)
    data_val_numeriques = data.select_dtypes(include=[int])
    #attribut = [f'Transaction_{i}' for i in range(1, 881)]
    attribut = [f'Transaction_{i}' for i in range(1, len(data) + 1)]
    data.index = attribut
except:
    st.write('Probleme de lecture des données')

st.sidebar.title(' Menu de navigation ')
menu = st.sidebar.selectbox('Sélectionner une option',['Accueil', 'Analyse Comparatives', 'Statistiques', 'Visualisation', 'Recommandations'])

if menu=='Accueil':
    st.subheader('Présentations des données de ventes')
    st.write("Voici un apercu des données de Beans & Pods sur l'exercice, ressortant des détails importants sur les choix, regions et modes caractérisant les transactions.")
    st.dataframe(data)
    st.write("Ces données seront soumises à des analyses desquelles ressortiront des résultats et recommandations")
    

elif menu == 'Analyse Comparatives':  
    st.subheader('Présentation des 50 premières transactions')
    st.dataframe(data.head(50))
    st.subheader('Calcul du nombre de transaction par Region')
    class_count = data.groupby('Region').size()
    st.write(class_count)
    st.write("La region avec la plus grosse activité est la region du SUD. La moins active est celle du CENTRE")

    st.subheader('Graphe de repartition des Regions')
    figure, ax_class = plt.subplots()
    data['Region'].value_counts().plot(kind='bar', color=['green', 'red', 'yellow'], ax= ax_class)
    ax_class.set_xlabel(' Niveau d activité des Region')
    ax_class.set_ylabel('Fréquence')
    st.pyplot(figure)

    st.subheader('Calcul du nombre de transaction par Mode d achat')
    channel_count = data.groupby('Channel').size()
    st.write(channel_count)
    st.write("Les Clients préfèrent acheter plus en magasin plutot qu'en ligne")

    st.subheader('Graphe de repartition des Modes d Achat')
    figure, ax_class = plt.subplots()
    data['Channel'].value_counts().plot(kind='bar', color=['green', 'red'], ax= ax_class)
    ax_class.set_xlabel('Mode de transaction (Online pour En ligne et Store En magasin)')
    ax_class.set_ylabel('Fréquence')
    st.pyplot(figure)

    st.subheader('Nombre de ventes par canal (Store vs Online)')
    vtes_channel = data.groupby('Channel').sum()
    st.write(vtes_channel[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']])

    st.subheader('Représentation graphique')
    vtes_channel[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']].plot(kind='bar', figsize=(10, 6))
    plt.title("Répartition des ventes par canal")
    plt.xlabel('Canal')
    plt.ylabel('Montant des ventes')
    st.pyplot(plt.gcf())

    st.subheader('Nombre de ventes par Region (Store vs Online)')
    vtes_region = data.groupby('Region').sum()
    st.write(vtes_region[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']])

    st.subheader('Représentation graphique')
    vtes_region[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']].plot(kind='bar', figsize=(10, 6))
    plt.title("Répartition des ventes par Region")
    plt.xlabel('Region')
    plt.ylabel('Montant des ventes')
    st.pyplot(plt.gcf())

    st.subheader("Les produits les plus vendus")
    best_ventes = data_val_numeriques.sum().sort_values()
    st.write(best_ventes)
    st.write("Le produit faisant le  plus de vente est le ROBUSTA et le moins est le CAPPUCINO ")

elif menu == 'Statistiques':  
    st.subheader('Statistiques descriptives')
    st.write(data.describe())

    st.subheader('La correlation de pearson')
    st.write(data_val_numeriques.corr())

    st.subheader('Analyse de certaines correlations')
    st.write(" Les produits ayant de corrélations trés élevées et élevées positives sont: l Arabica et l expresso; Espresso et Latte; Arabica et Latte")
    st.write(" Les produits ayant de moyennes et faibles corrélation positives sont: le Robusta et le Lungo; Espresso et Cappuccino")
    st.write(" Les produits ayant aucune corrélation sont: le Latte et le Cappuccino")
    st.write(" On a aussi des corrélations negatives: Robusta et Latte; Lungo et Latte")
    
   
elif menu == 'Visualisation':
  st.subheader('Les Histogrammes')
  data.hist(bins=20, figsize=(15, 10))
  st.pyplot(plt.gcf())

  st.subheader('Histogramme des grains Robusta (produit le plus vendu)')
  figure, ax_class = plt.subplots()
  ax_class.hist(data['Robusta'], color='blue')
  ax_class.set_xlabel('Robusta')
  ax_class.set_ylabel('Fréquence')
  st.pyplot(figure)

  st.subheader('Graphe de densité')
  data.plot(kind='density', subplots=True, sharex= False, layout=(3,3), figsize=(15, 15))
  st.pyplot(plt.gcf())

  st.subheader('Boites a moustaches')
  data.plot(kind='box', subplots=True, sharex= False, layout=(3,3), figsize=(15, 15))
  st.pyplot(plt.gcf())

  st.subheader('Matrice de corrélation')
  figure, ax_corr= plt.subplots(figsize=(10, 10))
  sns.heatmap(data_val_numeriques.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
  st.pyplot(figure)

  st.subheader('Scatter Matrix')
  scatter_matrix(data, figsize=(25, 25), c='g')
  st.pyplot(plt.gcf())

  st.subheader('Graphique des pairplots')
  sns.pairplot(data, hue='Robusta')
  st.pyplot(plt.gcf())   

elif menu == 'Recommandations':  
    
    st.subheader('Recommandations basées sur l’analyse des ventes')

    st.write("""
    ### 1. Maximiser les ventes en ligne
    Selon les comparaison des ventes par canal, les ventes en ligne sont très basses par rapport à celles en magasin. Il faut organiser des campagnes de marketing et renforcer le service en ligne pour élargir le marché sur Internet.
    """)

    st.write("""
    ### 2. Remonter les ventes de Cappucino
    Le Cappucino est le produit qui fait le moins de ventes, il est très en dessous par rapport aux autres produits.
    Nous recommandons d'offrir des promotions spéciales pour inciter les clients à acheter le Cappucino et le decouvrir.
    """)

    st.write("""
    ### 3. Comprendre les tendances régionales
    La region du Sud domine largement le classement des ventes, loin devant le Centre et le Nord. Je pense que 
    l'entreprise devrait s'activer dans cette zone, en multipliant les campagnes et actions publicitaires pour augmenter ses ventes dans ces zones 
    """)

    st.write("""
    ### 4. Collecter plus de données sur les préférences des clients
    Pour affiner les stratégies de marketing, Beans & Pods pourrait envisager de collecter des données supplémentaires sur les préférences des clients, telles que leurs types de produits préférés, la fréquence d'achat, et les heures de la journée où ils achètent. Cela permettra d'ameliorer le service a la clientele et eviter l'indisponibilité des produits .
    """)

    st.write("""
    ### 5. Améliorer la segmentation des clients
    Une segmentation plus fine des clients, basée sur des critères démographiques et comportementaux, pourrait également être bénéfique. Cela permettra de proposer des offres adaptées à différents segments de clientèle et d’augmenter les chances de conversion.
    """)