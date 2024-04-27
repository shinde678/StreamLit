import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go

st.title("Data Analysis Application")
st.subheader("This is simple data anylysis app.")

# Create a dropdown list for available datasets
dataset_list = ["iris", "titanic", "tips", "diamonds"]
selected_dataset = st.selectbox("Select a dataset", dataset_list)

# Load the selected dataset
if selected_dataset == "iris":
    df = sns.load_dataset("iris")
elif selected_dataset == "titanic":
    df = sns.load_dataset("titanic")
elif selected_dataset == "tips":
    df = sns.load_dataset("tips")
elif selected_dataset == "diamonds":
    df = sns.load_dataset("diamonds")    

# Provide an option to upload a custom dataset
if st.button("Upload custom dataset"):
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Display the selected dataset
st.write("## Selected Dataset")
st.write(df)

st.write("Number of rows", df.shape[0])
st.write("Number of Columns", df.shape[1])

# display column name with data type
st.write("Column Names and data types", df.dtypes)

# print the null values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Null values', df.isnull().sum().sort_values(ascending=False))


st.write(df.describe())


# print chart
x_axis = st.selectbox("Select X-axis", df.columns) 
y_axis = st.selectbox("Select Y-axis", df.columns) 

plot_type = st.selectbox("Select plot type", ['line', 'scatter', 'bar', 'hist', 'box', 'kde'])

# plot the data
if plot_type == 'line':
    st.line_chart(df [[x_axis, y_axis]])

elif plot_type == 'scatter':
    st.scatter_chart (df [[x_axis, y_axis]])

elif plot_type == 'bar':
    st.bar_chart(df[[x_axis, y_axis]])

elif plot_type == 'hist':
    df[x_axis].plot(kind='hist')
    st.pyplot()

elif plot_type == 'box':
    df[[x_axis, y_axis]].plot(kind='box')
    st.pyplot()
    
elif plot_type == 'kde':
    df[[x_axis, y_axis]].plot(kind='kde')
    st.pyplot()
    
    
# Create a heatmap
st.subheader('Heatmap')
# select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes (include=np.number).columns
corr_matrix = df [numeric_columns].corr()
numeric_columns = df.select_dtypes (include=np.number).columns
corr_matrix = df [numeric_columns].corr()

# Create a heatmap

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go. Figure(data=go. Heatmap(z=corr_matrix.values,
x=corr_matrix.columns,
y=corr_matrix.columns,
colorscale='Viridis'))
st.plotly_chart (heatmap_fig)
