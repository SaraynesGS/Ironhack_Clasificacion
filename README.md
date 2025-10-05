![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)
# Proyecto 3 | Modelo Supervisado de Clasificación
## Predicción de ataque al corazón (End-to-end)

---

### Objetivo
El objetivo de este proyecto es construir un modelo de clasificación supervisada (binaria o multiclase) utilizando un conjunto de datos elegido por nosotros. Trabajaremos con el dataset sobre predicción de ataques cardíacos para distinguir la presencia de enfermedad cardíaca entre los pacientes.

### Fuente de Datos
- **Dataset**: [Heart Attack Prediction](https://www.kaggle.com/datasets/imnikhilanand/heart-attack-prediction)
- **Descripción**: Este dataset contiene 76 atributos; para experimentos previos se ha utilizado un subconjunto de 14 variables. Usaremos las variables del dataset procesado conocido como "Cleveland database" que tiene un interés particular por su uso común en investigaciones de machine learning.

### Descripción de las Variables
1. **age**: Edad en años
2. **sex**: Sexo (1 = masculino; 0 = femenino)
3. **cp**: Tipo de dolor de pecho
4. **trestbps**: Presión arterial en reposo (mm Hg)
5. **chol**: Colesterol sérico en mg/dl
6. **fbs**: Nivel de azúcar en ayunas (> 120 mg/dl)
7. **restecg**: Resultados de electrocardiograma en reposo
8. **thalach**: Frecuencia cardíaca máxima alcanzada
9. **exang**: Angina provocada por el ejercicio (1 = sí; 0 = no)
10. **oldpeak**: Depresión ST inducida por el ejercicio
11. **slope**: Pendiente del segmento ST
12. **ca**: Número de vasos principales coloreados por fluoroscopia
13. **thal**: Defecto cardíaco
14. **num**: Diagnóstico de enfermedad cardíaca

### Fases del Proyecto

1. **Obtención de Datos**:
   - Elegimos el dataset sobre predicción de ataques cardíacos de Kaggle.

2. **Exploración Inicial**:
   - Cargamos los datos en un DataFrame y exploramos las primeras filas, dimensiones, y tipos de columnas.

3. **Limpieza de Datos**:
   - Tratamos valores nulos, eliminamos duplicados y verificamos inconsistencias.

4. **Ingeniería de Características**:
   - Creamos nuevas variables según fuera necesario y aseguramos que todas las variables fueran numéricas mediante hot-encoding y modificación de dtype.

5. **Análisis Exploratorio de Datos (EDA)**:
   - Analizamos las variables predictoras y target, e incluimos gráficos descriptivos.

    <img width="963" height="692" alt="Captura de pantalla 2025-10-04 a la(s) 22 14 23" src="https://github.com/user-attachments/assets/bad6fc87-89c5-4c8b-a869-ec9b98b14c06" />

    <img width="963" height="698" alt="Captura de pantalla 2025-10-04 a la(s) 22 13 57" src="https://github.com/user-attachments/assets/fd641c24-8707-44b0-9062-2f360c67f687" />

6. **Selección de Características**:
   - Elegimos columnas para el modelo y codificamos las variables categóricas.

7. **Preparación para el Modelado**:
   - División de datos X (predictoras) y y (target), aplicación de transformations (normalización y estandarización).

8. **Entrenamiento y Validación**:
   - Realizamos un train-test split (80-20) y entrenamos diferentes modelos: regresión logística, KNN, árbol de decisión, y random forest.

9. **Evaluación y Comparación de Modelos**:
   - Calculamos métricas de evaluación como accuracy, precision, recall, F1 score y curva ROC. Comparamos los resultados en una tabla.
  
     <img width="331" height="265" alt="Captura de pantalla 2025-10-04 a la(s) 22 14 44" src="https://github.com/user-attachments/assets/766e6868-3a72-45ef-8196-4e171b98e613" />

10. **Importancia de Características**:
    - Evaluamos la importancia de las características mediante análisis del modelo.

11. **Ajuste de Hiperparámetros**:
    - Realizamos la optimización de hiperparámetros para mejorar el rendimiento.
      <img width="551" height="182" alt="Captura de pantalla 2025-10-04 a la(s) 22 15 48" src="https://github.com/user-attachments/assets/a621f802-2195-4e5a-bc30-226576242bc4" />

      <img width="398" height="152" alt="Captura de pantalla 2025-10-04 a la(s) 22 15 28" src="https://github.com/user-attachments/assets/9155d36b-6689-4f6e-8b69-2c8a9e6ebd60" />


### Conclusiones
1. **Parámetros Óptimos**: El modelo de *Random Forest* alcanzó un rendimiento óptimo con parámetros seleccionados mediante tuning: bootstrap: True, max_depth: 20, max_features: 'sqrt', min_samples_leaf: 2, min_samples_split: 2, y n_estimators: 600, logrando una accuracy en validación cruzada de 0.7989.

2. **Rendimiento del Modelo**: En el conjunto de prueba, el modelo alcanzó una accuracy del 84%, con una precision de 87% para la clase 0 y 80% para la clase 1, un recall de 83% y 84% respectivamente, y un f1-score de 0.84, mostrando un buen equilibrio entre precision y recall.

3. **Importancia de las Características**: Las variables *Depresión ST inducida por el ejercicio* y *Colesterol sérico en mg/dl* fueron identificadas como factores cruciales para la predicción de enfermedades cardíacas, alineándose con la literatura médica existente.

---
