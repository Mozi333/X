import streamlit as st
import pandas as pd

# add a header title to the app
st.header('PLAYER SCOUT 🕵️‍♀️')

# load the Excel file using pandas
excel_file = st.file_uploader('Upload an Excel file:')
df = pd.read_excel(excel_file)

# add a title to the sidebar menu
st.sidebar.title('Filter players')

# create a sidebar menu with filters for "Age" and "Minutes played"
age_filter = st.sidebar.slider('Age', df['Age'].min(), df['Age'].max(), (df['Age'].min(), df['Age'].max()))
minutes_filter = st.sidebar.slider('Minutes played', df['Minutes played'].min(), df['Minutes played'].max(), (df['Minutes played'].min(), df['Minutes played'].max()))

# apply the filters to the original DataFrame
df = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) & (df['Minutes played'] >= minutes_filter[0]) & (df['Minutes played'] <= minutes_filter[1])]

# add a subheader to the app
st.subheader('Select metrics')
# create a multiselect menu with all the non-numeric columns
selected_non_numeric_columns = st.multiselect('Select the non-numeric columns to include:', df.select_dtypes(exclude='number').columns)

# create a multiselect menu with all the numeric columns
selected_numeric_columns = st.multiselect('Select the numeric columns to standardize:', df.select_dtypes('number').columns)

# add a subheader to the app
st.subheader('Choose weights given to each metric selected')
# create an input field for each selected numeric column where the user can specify a weight
column_weights = {}
for col in selected_numeric_columns:
  column_weights[col] = st.number_input(f'Weight for {col}:', min_value=0.0, max_value=1.0, value=1.0 / len(selected_numeric_columns))

# standardize the selected numeric columns and create new columns with the standardized values
for col in selected_numeric_columns:
  df[col + '_std'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# create a new DataFrame with the standardized numeric columns, the selected non-numeric columns, and a new column for the average of the standardized numeric columns
std_df = df[selected_numeric_columns + [col + '_std' for col in selected_numeric_columns] + selected_non_numeric_columns]
std_df['Ranking'] = 0

# apply the weights specified by the user to the standardized numeric columns
for col in selected_numeric_columns:
  std_df['Ranking'] += column_weights[col] * std_df[col + '_std']


# standardize the "Ranking" column and sort the DataFrame by that column
std_df['Ranking'] = (std_df['Ranking'] - std_df['Ranking'].min()) / (std_df['Ranking'].max() - std_df['Ranking'].min())


# reorder the columns so that the non-numeric columns appear first
std_df = std_df[['Ranking'] + selected_non_numeric_columns + selected_numeric_columns + [col + '_std' for col in selected_numeric_columns]]

# add a subheader to the app
st.subheader('Top players based on selected statistics')

# round all the numeric columns to 2 decimal places
std_df = std_df.round(2)

# apply a function to all the elements in the DataFrame to format the numeric values as strings with only up to 2 decimal places
std_df = std_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

# display the new DataFrame
st.dataframe(std_df)







