import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn import metrics

st.title('Fetal Health Predictor')
st.write('We are trying to understand what indicates normal, suspect, or pathological infant health based on cardiotocogram results, which will help medical professionals take the appropriate action to reduce child mortality rates.')


# our produciton model
df = pd.read_csv('./fetal_health.csv')

df.rename(columns={'baseline value' : 'baseline_value'}, inplace=True)

X = df.loc[:, 'baseline_value': 'mean_value_of_long_term_variability']

y = df['fetal_health']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y )

ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)

X_test_sc = ss.transform(X_test)

#instantiate model
gbc = GradientBoostingClassifier(learning_rate=.1, max_depth=2, min_samples_leaf=2, n_estimators=500)

#fit to scaled data
gbc.fit(X_train_sc, y_train)

#inputs from user
baseline_value = st.number_input('Baseline Value', min_value=100, max_value=200, step=1)
accelerations = st.number_input('Accelerations', min_value=0.00, max_value=.02, step=.01)
fetal_movement = st.number_input('Fetal Movement', min_value=0.00, max_value=.50, step=.01)
uterine_contractions = st.number_input('Uterine Contractions', min_value=0.00, max_value=.02, step=.01)
light_decelerations = st.number_input('Light Decelerations', min_value=0.00, max_value=.02, step=.01)
severe_decelerations = st.number_input('Severe Decelerations', min_value=0.00, max_value=.02, step=.01)
prolongued_decelerations = st.number_input('Prolongued Decelerations', min_value=0.00, max_value=.02, step=.01)
abnormal_short_term_variability = st.number_input('Percentage of Time with Abnormal Short Term Variability', min_value=0, max_value=100, step=1)
mean_value_of_short_term_variability = st.number_input('Mean Value Short Term Variability', min_value=0.0, max_value=10.0, step=.1)
percentage_of_time_with_abnormal_long_term_variability = st.number_input('Percentage of Time with Abnormal Long Term Variability', min_value=0, max_value=100, step=1)
mean_value_of_long_term_variability = st.number_input('Mean Value Long Term Variability', min_value=0.0, max_value=100.0, step=.1)


inputs = [[baseline_value, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations,
abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability]]

inputs_sc = ss.transform(inputs)

if st.button('Predict'):
    #make predictions
    predictions = gbc.predict(inputs_sc)

    if predictions[0] == 1:
        st.write("The patient's fetal health is Normal")
    elif predictions[0] == 2:
        st.write("The patient's fetal health is Suspect")
    else:
        st.write("The patient's fetal health is Pathological")
