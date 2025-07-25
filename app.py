# example of ML APP with streamlit

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load the data
df, target_name = load_data()

# Instantiate the RandomForest model
model = RandomForestClassifier()

# Train the model on the entire dataset (excluding the species column as the target)
X = df.iloc[:, :-1]  # Features (all columns except 'species')
y = df['species']    # Target (species column)

# Fit the model
model.fit(X, y)

# Debug: Print the shape of the features and target
print(X.shape)  # Should be (number_of_rows, number_of_features)
print(y.shape)  # Should be (number_of_rows,)

# Streamlit sidebar for user input
st.sidebar.title('Input Features')

sepal_length = st.sidebar.slider('sepal length',
                                 float(df['sepal length (cm)'].min()),
                                 float(df['sepal length (cm)'].max()))

sepal_width = st.sidebar.slider('sepal width',
                                float(df['sepal width (cm)'].min()),
                                float(df['sepal width (cm)'].max()))

petal_length = st.sidebar.slider('petal length',
                                 float(df['petal length (cm)'].min()),
                                 float(df['petal length (cm)'].max()))

petal_width = st.sidebar.slider('petal width',
                                float(df['petal width (cm)'].min()),
                                float(df['petal width (cm)'].max()))

# Combine the input values into one list for prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction
prediction = model.predict(input_data)
predicted_species = target_name[prediction[0]]

# Display result
st.write('Prediction:')
st.write(f'The predicted species is: {predicted_species}')
