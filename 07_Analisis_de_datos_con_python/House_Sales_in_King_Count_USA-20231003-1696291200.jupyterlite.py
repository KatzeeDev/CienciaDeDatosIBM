#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="300" alt="Skills Network Logo">
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Final Project: House Sales in King County, USA </font></h1>
# 

# <h2>Table of Contents</h2>
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ul>
#     <li><a href="#Instructions">Instructions</a></li>
#     <li><a href="#About-the-Dataset">About the Dataset</a></li>
#     <li><a href="#Module-1:-Importing-Data-Sets">Module 1: Importing Data </a></li>
#     <li><a href="#Module-2:-Data-Wrangling">Module 2: Data Wrangling</a> </li>
#     <li><a href="#Module-3:-Exploratory-Data-Analysis">Module 3: Exploratory Data Analysis</a></li>
#     <li><a href="#Module-4:-Model-Development">Module 4: Model Development</a></li>
#     <li><a href="#Module-5:-Model-Evaluation-and-Refinement">Module 5: Model Evaluation and Refinement</a></li>
# </a></li>
# </div>
# <p>Estimated Time Needed: <strong>75 min</strong></p>
# </div>
# 
# <hr>
# 

# # Instructions
# 

# In this assignment, you are a Data Analyst working at a Real Estate Investment Trust. The Trust would like to start investing in Residential real estate. You are tasked with determining the market price of a house given a set of features. You will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on. This is a template notebook; your job is to complete the ten questions. Some hints to the questions are given.
# 
# As you are completing this notebook, take and save the **screenshots** of the final outputs of your solutions (e.g., final charts, tables, calculation results etc.). They will need to be shared in the following Peer Review section of the Final Project module.
# 

# # About the Dataset
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. It was taken from [here](https://www.kaggle.com/harlfoxem/housesalesprediction?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-wwwcourseraorg-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01). It was also slightly modified for the purposes of this course. 
# 

# | Variable      | Description                                                                                                 |
# | ------------- | ----------------------------------------------------------------------------------------------------------- |
# | id            | A notation for a house                                                                                      |
# | date          | Date house was sold                                                                                         |
# | price         | Price is prediction target                                                                                  |
# | bedrooms      | Number of bedrooms                                                                                          |
# | bathrooms     | Number of bathrooms                                                                                         |
# | sqft_living   | Square footage of the home                                                                                  |
# | sqft_lot      | Square footage of the lot                                                                                   |
# | floors        | Total floors (levels) in house                                                                              |
# | waterfront    | House which has a view to a waterfront                                                                      |
# | view          | Has been viewed                                                                                             |
# | condition     | How good the condition is overall                                                                           |
# | grade         | overall grade given to the housing unit, based on King County grading system                                |
# | sqft_above    | Square footage of house apart from basement                                                                 |
# | sqft_basement | Square footage of the basement                                                                              |
# | yr_built      | Built Year                                                                                                  |
# | yr_renovated  | Year when house was renovated                                                                               |
# | zipcode       | Zip code                                                                                                    |
# | lat           | Latitude coordinate                                                                                         |
# | long          | Longitude coordinate                                                                                        |
# | sqft_living15 | Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area |
# | sqft_lot15    | LotSize area in 2015(implies-- some renovations)                                                            |
# 

# ## **Import the required libraries**
# 

# In[ ]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"


# In[ ]:


# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:


#!pip install -U scikit-learn


# In[ ]:


# Importar las librerías necesarias para el análisis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Module 1: Importing Data Sets
# 

# Download the dataset by running the cell below.
# 

# In[ ]:


# Función para descargar datos desde URL (adaptada para entorno local)
import urllib.request

def descargar_datos(url, nombre_archivo):
    try:
        urllib.request.urlretrieve(url, nombre_archivo)
        print(f"Datos descargados exitosamente como {nombre_archivo}")
    except Exception as e:
        print(f"Error al descargar: {e}")


# In[ ]:


# URL del conjunto de datos
ruta_archivo = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'


# In[ ]:


# Descargar el conjunto de datos
descargar_datos(ruta_archivo, "ventas_casas.csv")
nombre_archivo = "ventas_casas.csv"


# Load the csv:
# 

# In[ ]:


# Cargar el conjunto de datos en un DataFrame
df = pd.read_csv(nombre_archivo)


# > Note: This version of the lab is working on JupyterLite, which requires the dataset to be downloaded to the interface.While working on the downloaded version of this notebook on their local machines(Jupyter Anaconda), the learners can simply **skip the steps above,** and simply use the URL directly in the `pandas.read_csv()` function. You can uncomment and run the statements in the cell below.
# 

# In[ ]:


# Método alternativo: cargar directamente desde URL (comentado)
# ruta_archivo = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
# df = pd.read_csv(ruta_archivo)


# We use the method <code>head</code> to display the first 5 columns of the dataframe.
# 

# In[ ]:


# Mostrar las primeras 5 filas del conjunto de datos
df.head()


# ### Question 1
# 
# Display the data types of each column using the function dtypes. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
# 

# In[ ]:


# Pregunta 1: Mostrar los tipos de datos de cada columna usando el atributo dtypes
print("Tipos de datos de cada columna:")
print(df.dtypes)


# We use the method describe to obtain a statistical summary of the dataframe.
# 

# In[ ]:


# Resumen estadístico del conjunto de datos
df.describe()


# # Module 2: Data Wrangling
# 

# ### Question 2
# 
# Drop the columns <code>"id"</code>  and <code>"Unnamed: 0"</code> from axis 1 using the method <code>drop()</code>, then use the method <code>describe()</code> to obtain a statistical summary of the data. Make sure the <code>inplace</code> parameter is set to <code>True</code>. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
# 

# In[ ]:


# Pregunta 2: Eliminar las columnas "id" y "Unnamed: 0" del eje 1 usando drop()
# Verificar qué columnas existen en el DataFrame
print("Columnas en el DataFrame:")
print(df.columns.tolist())

# Eliminar columnas innecesarias (si existen)
columnas_a_eliminar = []
if 'id' in df.columns:
    columnas_a_eliminar.append('id')
if 'Unnamed: 0' in df.columns:
    columnas_a_eliminar.append('Unnamed: 0')

if columnas_a_eliminar:
    df.drop(columnas_a_eliminar, axis=1, inplace=True)
    print(f"Columnas eliminadas: {columnas_a_eliminar}")
else:
    print("No se encontraron las columnas 'id' o 'Unnamed: 0' para eliminar")

# Mostrar resumen estadístico después de eliminar columnas
print("\nResumen estadístico después de eliminar columnas:")
df.describe()


# We can see we have missing values for the columns <code> bedrooms</code>  and <code> bathrooms </code>
# 

# In[ ]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# We can replace the missing values of the column <code>'bedrooms'</code> with the mean of the column  <code>'bedrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code>inplace</code> parameter to <code>True</code>
# 

# In[ ]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# We also replace the missing values of the column <code>'bathrooms'</code> with the mean of the column  <code>'bathrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code> inplace </code>  parameter top <code> True </code>
# 

# In[ ]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[ ]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # Module 3: Exploratory Data Analysis
# 

# ### Question 3
# 
# Use the method <code>value_counts</code> to count the number of houses with unique floor values, use the method <code>.to_frame()</code> to convert it to a data frame. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
# 

# In[ ]:


# Pregunta 3: Usar value_counts para contar casas con valores únicos de pisos y convertir a DataFrame
recuento_pisos = df['floors'].value_counts().to_frame()
print("Recuento de casas por número de pisos:")
print(recuento_pisos)


# ### Question 4
# 
# Use the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers. Take a screenshot of your code and boxplot. You will need to submit the screenshot for the final project. 
# 

# In[ ]:


# Pregunta 4: Boxplot para determinar valores atípicos de precios según vista al mar
plt.figure(figsize=(10, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('Distribución de Precios: Casas con y sin Vista al Mar')
plt.xlabel('Vista al Mar (0 = No, 1 = Sí)')
plt.ylabel('Precio')
plt.show()


# ### Question 5
# 
# Use the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price. Take a screenshot of your code and scatterplot. You will need to submit the screenshot for the final project. 
# 

# In[ ]:


# Pregunta 5: Regplot para determinar correlación entre sqft_above y precio
plt.figure(figsize=(10, 6))
sns.regplot(x='sqft_above', y='price', data=df)
plt.title('Correlación entre Área sobre el Sótano (sqft_above) y Precio')
plt.xlabel('Área sobre el Sótano (pies cuadrados)')
plt.ylabel('Precio')
plt.show()


# We can use the Pandas method <code>corr()</code>  to find the feature other than price that is most correlated with price.
# 

# In[ ]:


# Encontrar la característica más correlacionada con el precio
df_numerico = df.select_dtypes(include=[np.number])
correlaciones_precio = df_numerico.corr()['price'].sort_values()
print("Correlaciones con el precio (ordenadas de menor a mayor):")
print(correlaciones_precio)


# # Module 4: Model Development
# 

# We can Fit a linear regression model using the  longitude feature <code>'long'</code> and  caculate the R^2.
# 

# In[ ]:


# Ejemplo: Modelo de regresión lineal usando la característica 'long'
X = df[['long']]
Y = df['price']
modelo_lineal = LinearRegression()
modelo_lineal.fit(X, Y)
r2_longitud = modelo_lineal.score(X, Y)
print(f"R² para el modelo con 'long': {r2_longitud:.4f}")


# ### Question  6
# 
# Fit a linear regression model to predict the <code>'price'</code> using the feature <code>'sqft_living'</code> then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
# 

# In[ ]:


# Pregunta 6: Modelo de regresión lineal para predecir precio usando 'sqft_living'
X_6 = df[['sqft_living']]
Y_6 = df['price']
modelo_6 = LinearRegression()
modelo_6.fit(X_6, Y_6)
r2_sqft_living = modelo_6.score(X_6, Y_6)
print(f"R² para el modelo con 'sqft_living': {r2_sqft_living:.4f}")


# ### Question 7
# 
# Fit a linear regression model to predict the <code>'price'</code> using the list of features:
# 

# In[ ]:


# Lista de características para el modelo múltiple
caracteristicas = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]


# Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
# 

# In[ ]:


# Pregunta 7: Modelo de regresión lineal usando múltiples características
X_7 = df[caracteristicas]
Y_7 = df['price']
modelo_7 = LinearRegression()
modelo_7.fit(X_7, Y_7)
r2_multiple = modelo_7.score(X_7, Y_7)
print(f"R² para el modelo con múltiples características: {r2_multiple:.4f}")


# ### This will help with Question 8
# 
# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 
# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# The second element in the tuple  contains the model constructor
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>
# 

# In[ ]:


# Lista de tuplas para crear el pipeline
entrada_pipeline = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]


# ### Question 8
# 
# Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list <code>features</code>, and calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
# 

# In[ ]:


# Pregunta 8: Crear pipeline para escalado, transformación polinómica y regresión lineal
pipeline_8 = Pipeline(entrada_pipeline)
X_8 = df[caracteristicas]
Y_8 = df['price']
pipeline_8.fit(X_8, Y_8)
r2_pipeline = pipeline_8.score(X_8, Y_8)
print(f"R² para el pipeline (escalado + polinómica + regresión): {r2_pipeline:.4f}")


# # Module 5: Model Evaluation and Refinement
# 

# Import the necessary modules:
# 

# In[ ]:


# Importar módulos necesarios para evaluación de modelos
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("Módulos importados correctamente")


# We will split the data into training and testing sets:
# 

# In[ ]:


# Dividir los datos en conjuntos de entrenamiento y prueba
caracteristicas = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]    
X = df[caracteristicas]
Y = df['price']

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(X, Y, test_size=0.15, random_state=1)

print("Número de muestras de prueba:", x_prueba.shape[0])
print("Número de muestras de entrenamiento:", x_entrenamiento.shape[0])


# ### Question 9
# 
# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
# 

# In[ ]:


# Importar Ridge regression
from sklearn.linear_model import Ridge


# In[ ]:


# Pregunta 9: Crear y entrenar modelo Ridge con parámetro de regularización 0.1
modelo_ridge = Ridge(alpha=0.1)
modelo_ridge.fit(x_entrenamiento, y_entrenamiento)
r2_ridge = modelo_ridge.score(x_prueba, y_prueba)
print(f"R² para regresión Ridge (α=0.1): {r2_ridge:.4f}")


# ### Question 10
# 
# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2. You will need to submit it for the final project.
# 

# In[ ]:


# Pregunta 10: Transformación polinómica de segundo orden + Ridge regression
# Crear transformador polinómico de segundo orden
transformador_poli = PolynomialFeatures(degree=2, include_bias=False)

# Transformar datos de entrenamiento y prueba
x_entrenamiento_poli = transformador_poli.fit_transform(x_entrenamiento)
x_prueba_poli = transformador_poli.transform(x_prueba)

# Crear y entrenar modelo Ridge con características polinómicas
modelo_ridge_poli = Ridge(alpha=0.1)
modelo_ridge_poli.fit(x_entrenamiento_poli, y_entrenamiento)

# Calcular R² usando datos de prueba
r2_ridge_poli = modelo_ridge_poli.score(x_prueba_poli, y_prueba)
print(f"R² para Ridge con transformación polinómica de 2° orden (α=0.1): {r2_ridge_poli:.4f}")


# <p>Once you complete your notebook you will have to share it. You can download the notebook by navigating to "File" and clicking on "Download" button.
#         <p><img width="600" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Module%206/images/DA0101EN_FA_Image21.png" alt="share notebook" style="display: block; margin-left: auto; margin-right: auto;"></p>
#         <p></p>
# <p>This will save the (.ipynb) file on your computer. Once saved, you can upload this file in the "My Submission" tab, of the "Peer-graded Assignment" section.  
#           
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Michelle Carey</a>, <a href="https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Mavis Zhou</a>
# 

# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# <!--## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By      | Change Description                           |
# | ----------------- | ------- | --------------- | -------------------------------------------- |
# | 2020-12-01        | 2.2     | Aije Egwaikhide | Coverted Data describtion from text to table |
# | 2020-10-06        | 2.1     | Lakshmi Holla   | Changed markdown instruction of Question1    |
# | 2020-08-27        | 2.0     | Malika Singla   | Added lab to GitLab                          |
# | 2022-06-13        | 2.3     | Svitlana Kramar | Updated Notebook sharing instructions        |
# | <hr>              |         |                 |                                              |
# 
# 
# --!>
# <p>
# 
