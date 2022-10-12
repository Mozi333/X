import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import math
import matplotlib.pyplot as plt
from highlight_text import fig_text
from adjustText import adjust_text
from PIL import Image
from urllib.request import urlopen
from adjustText import adjust_text
from highlight_text import fig_text
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import plotly.express as px
import altair as alt
import plotly.express as px
pd.set_option('mode.use_inf_as_na', True)
import requests
import openpyxl
from pathlib import Path
from mplsoccer import PyPizza, add_image, FontManager

#-------- hide hamburger menu and made with streamlit text
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#-----------------------------FUNCTIONS--------------------------------------------------------------

st.title('SCOUTING APP 🕵🏼‍♂️')



def load_data():
    
    data = (r'https://github.com/Mozi333/X/blob/main/delanterosamericas1.xlsx?raw=true')
    file = requests.get(data)
    df = pd.read_excel(file.content)
    
    #add commas to player values
    #df['Valor de mercado'] = df['Valor de mercado'].apply('{:,}'.format)
    
    #edit identical strings in name colum
    num = df.groupby('Player').cumcount()
    df.loc[num.ne(0), 'Player'] += ' '+num.astype(str)
    
    #convert nan to 0
    df.fillna(0, inplace=True)
    
    return df

def new_metrics(df):  
    
    #create Goles x remate metric
    df["Goal %"] = round(df['Non-penalty goals'] / df['Shots'], 2) 
    
    #goal ratio
    df['Goal Ratio'] = round(df['Shots'] / df['Non-penalty goals'], 2)
    
    #Create new column 90 min played
    df['90s'] = df['Minutes played'] / 90
    df['90s'] = df['90s'].round()
    
    #Create column with penalty xG
    df["penalty_xG"] = df['Penalties taken'] * 0.76 
    
    #Create column with  npxG
    df["nonpenalty_xG"] = round(df['xG'] - df["penalty_xG"], 2) 
    
    #Create column with pxG per 90
    df["penalty_xG/90"] = round(df['penalty_xG'] / df["90s"], 2) 
    
    #Create column with  npxG per 90
    df["nonpenalty_xG/90"] = round(df['xG per 90'] - df["penalty_xG/90"], 2) 
    
    #Create column with  xG and npG per 90
    df["Sum_xGp90_and_Goalsx90"] = round(df['nonpenalty_xG/90'] + df["Non-penalty goals per 90"], 2)

    #Create column with  xA and Assist per 90
    df["Sum_xAx90_and_Assistx90"] = round(df['xA per 90'] + df["Assists per 90"], 2) 
    
    
    #goal difference from xG p90
    df["xG_Difference"] = round(df['Non-penalty goals per 90'] - df['nonpenalty_xG/90'], 2) 

    
#Dividir Playeres por posicion 

def get_position():

    df['Main_position'] = df['Position'].apply(lambda x: x.split(',')[0])

    # create a list of our conditions
    conditions = [(df['Position'] == 'CF') | (df['Position'] == 'LWF') | (df['Position'] == 'RWF'),
        (df['Position'] == 'RW') | (df['Position'] == 'LF') | (df['Position'] == 'RAMF') |
        (df['Position'] == 'LAMF') | (df['Position'] == 'AMF') | (df['Position'] == 'RCMF') |
        (df['Position'] == 'LCMF') | (df['Position'] == 'RDMF') | (df['Position'] == 'LDMF') |
        (df['Position'] == 'LWB') | (df['Position'] == 'RWB') | (df['Position'] == 'DMF') | (df['Position'] == 'LW'),
        (df['Position'] == 'LB') | (df['Position'] == 'RCB') | (df['Position'] == 'LCB') |
        (df['Position'] == 'RB') | (df['Position'] == 'RWB') | (df['Position'] == 'LWB') | (df['Position'] == 'CB'),
        (df['Position'] == 'GK')]

    # create a list of the values we want to assign for each condition
    values = ['Striker', 'Midfield', 'Defense', 'Goalkeeper']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['Field_position'] = np.select(conditions, values)
    
#result all 3 aspects ------------Style Index Colors------------------

def styler(v):
    if v > .84:
        return 'background-color:#3498DB' #blue
    elif v > .60:
        return 'background-color:#45B39D' #green
    if v < .15:
        return 'background-color:#E74C3C' #red
    elif v < .40:
        return 'background-color:#E67E22' #orange
    else:
        return 'background-color:#F7DC6F'  #yellow
    
#---------------------------start---------------------

uploaded_file = st.file_uploader("Load Wyscout database")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    df = load_data()
    
#---------New Metrics ------------------------------------

new_metrics(df)
get_position()

#Creates new columns with total metrics (Metrics p90 x 90s played)
df = df.join(df.filter(like='per 90').apply(lambda x: x.mul(df['90s'])).round().rename(columns=lambda x: x.replace('per 90', ' Total')))


df.fillna(0, inplace=True)

#replace infinity to 0
df['Goal %'].replace(np.inf, 0, inplace=True)
df['Goal Ratio'].replace(np.inf, 0, inplace=True)



#--------------------------------------------------------PRINT RAW DATA TABLE-----------------------

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')


#convert data to str to avoid errors
#df = df.astype(str)


# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.dataframe(df)

#-------------------------------------------------------------------------------filter players----------------------


st.sidebar.title('Filters')

#age
st.sidebar.write('Filter players by age:')
age = st.sidebar.slider('Age:', 0, 45, 40)

#Minutes Played
st.sidebar.write('Filter players by minutes played:')
minutes = st.sidebar.slider('Minutes:', 0, 5000, 200)

#xG
st.sidebar.write('Filter players by xG per 90 minutes played:')
xGp90 = st.sidebar.slider('xG:', 0.0, 1.5, 0.0)

df = df.loc[(df['Minutes played'] > minutes) & (df['Age'] < age) & (df['Position'] != 'GK') & ~(df['nonpenalty_xG/90'] < xGp90) ]
df.Player.unique()

#-------ASSIGNT VALUES FOR LATER USE------------------

#Assign Player name value for filter
name = df['Player']

# ------------------------------------------------USER INPUT METRICS--------------------


'## CHOOSE METRICS TO CREATE CUSTOM TABLE'
cols = st.multiselect('Metrics:', df.columns, default=['Player', 'Team within selected timeframe', 'Age', 'Minutes played',  
                                                        'Passport country', 'xG', 'Goals', 'Non-penalty goals', 'Goal %', 'Shots', 
                                                        'Assists', 'xA per 90', 'xA'])


# show dataframe with the selected columns
st.write(df[cols])
