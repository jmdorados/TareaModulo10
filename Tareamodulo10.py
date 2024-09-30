import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuración de la página
st.set_page_config(page_title='Tarea Módulo 10', layout='wide')

st.title('Bienvenido a mi tarea del Módulo 10')
st.sidebar.title('Integrantes')
st.sidebar.success('1. Nelson Estrada')
st.sidebar.success('2. Santiago Gallardo ')
st.sidebar.success('3. Jose Mario Dorado')

# Lista de opciones
#opciones = ['Cargar datos']

# Seleccionar una opción
#opcion = st.sidebar.selectbox('Seleccione una opción', opciones)

@st.cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.xlsx'):
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado. Solo se aceptan archivos CSV y XLSX.")
        return df
    else:
        return None

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster')

st.title('Análisis de Clustering y PCA en el Maratón de Nueva York')

# Subir el archivo CSV
file = st.file_uploader("Subir archivo CSV", type='csv')

if file is not None:
    data = pd.read_csv(file)

    # **Descripción del dataset**
    st.subheader('Descripción del dataset')
    st.write(f'El dataset tiene {data.shape[0]} filas y {data.shape[1]} columnas.')
    st.write('Las primeras filas del dataset son:')
    st.write(data.head())
  
    # **Revisión de tipos de datos**
    st.subheader('Revisión de tipos de datos')
    st.write(data.dtypes)

    # **Datos nulos**: Contar valores nulos en cada columna
    st.subheader('Datos Nulos')
    st.write(data.isnull().sum())

    # **Análisis estadístico descriptivo**: Mostrar estadísticas básicas
    st.subheader('Análisis Estadístico Descriptivo')
    st.write(data.describe())

    # **Análisis de distribución**: Mostrar histogramas para las variables numéricas
    st.subheader('Distribución de las variables numéricas')
    numeric_columns = ['place', 'age', 'time']
    fig, ax = plt.subplots(1, len(numeric_columns), figsize=(18, 5))
    for i, col in enumerate(numeric_columns):
        sns.histplot(data[col], kde=True, ax=ax[i])
        ax[i].set_title(f'Distribución de {col}')
    st.pyplot(fig)

    # **Valores únicos en columnas categóricas**
    st.subheader('Valores únicos en columnas categóricas')
    st.write('Columna `gender`:', data['gender'].unique())
    st.write('Columna `home`:', data['home'].nunique(), 'valores únicos')

    # **Mapa de calor de correlación entre variables numéricas**
    st.subheader('Mapa de calor de correlación entre variables numéricas')
    corr = data[['place', 'age', 'time']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # **Convertir 'gender' a formato numérico**: 0 = Male, 1 = Female
    data['gender'] = data['gender'].apply(lambda x: 0 if x == 'Male' else 1)

    # **Filtrar filas donde 'home' tiene exactamente 2 caracteres** (representan estados de USA)
    data_filtered = data[data['home'].apply(lambda x: len(x) == 2)].copy()

    # **One-Hot Encoding para el campo 'home'**
    data_filtered = pd.get_dummies(data_filtered, columns=['home'])

    # **Selección de columnas numéricas** para el análisis no supervisado
    numeric_columns = ['place', 'age', 'time', 'gender'] + [col for col in data_filtered.columns if 'home_' in col]

    # **Normalización Min-Max** para clustering
    min_max_scaler = MinMaxScaler()
    data_min_max_scaled = data_filtered.copy()
    data_min_max_scaled[numeric_columns] = min_max_scaler.fit_transform(data_filtered[numeric_columns])

    # **Normalización Z-score** para PCA
    z_score_scaler = StandardScaler()
    data_z_score_scaled = data_filtered.copy()
    data_z_score_scaled[numeric_columns] = z_score_scaler.fit_transform(data_filtered[numeric_columns])

    # **Clustering**: Usar K-Means en los datos normalizados con Min-Max Scaling
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    data_min_max_scaled['cluster'] = kmeans.fit_predict(data_min_max_scaled[numeric_columns])

    # **Reducción de dimensionalidad con PCA**
    pca = PCA(n_components=2)
    data_z_score_scaled_pca = pca.fit_transform(data_z_score_scaled[numeric_columns])

    # **Añadir componentes PCA** a los datos originales
    data_z_score_scaled['PCA1'] = data_z_score_scaled_pca[:, 0]
    data_z_score_scaled['PCA2'] = data_z_score_scaled_pca[:, 1]

    # **Mostrar resultados del clustering**
    st.subheader('Resultados de K-Means Clustering (Min-Max Scaling)')
    st.write(data_min_max_scaled[['place', 'age', 'time', 'gender', 'cluster'] + [col for col in data_min_max_scaled.columns if 'home_' in col]].head())

    # **Mostrar resultados de PCA**
    st.subheader('Resultados de PCA (Z-score Scaling)')
    st.write(data_z_score_scaled[['PCA1', 'PCA2']].head())

    # **Visualización de Clusters con PCA**
    st.subheader('Visualización de Clusters con PCA')
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_z_score_scaled['PCA1'], data_z_score_scaled['PCA2'], c=data_min_max_scaled['cluster'], cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    st.pyplot(fig)

    # **Descargar dataset con clusters y componentes PCA**
    st.subheader('Descargar archivo con Clusters y PCA')
    data_z_score_scaled['cluster'] = data_min_max_scaled['cluster']
    csv_download = data_z_score_scaled.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV",
        data=csv_download,
        file_name='maraton_clusters_pca.csv',
        mime='text/csv'
    )
else:
    st.write("Por favor, sube un archivo CSV para comenzar el análisis.")
