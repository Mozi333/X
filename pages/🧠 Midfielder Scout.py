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
import time


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

st.title('MIDFIELDER SCOUT 🕵🏼‍♂️🧠')


def load_data():
    
    data = (r'https://github.com/Mozi333/X/blob/main/data/mediosbeuropa.xlsx?raw=true')
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
    
    #Shots minus penalties
    
    df['non_penalty_shots'] = df['Shots'] - df['Penalties taken']
    
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
    
    #xG per shot average
    df['np_xG_per_shot_average'] =  df['nonpenalty_xG'] / df['non_penalty_shots']

    
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

    
#-------------------------------------------------------------------------------filter players----------------------


st.sidebar.title('Set Filters')

#age
st.sidebar.write('Filter players by age:')
age = st.sidebar.slider('Age:', 0, 45, 40)


#Minutes Played
st.sidebar.write('Filter players by minutes played:')
minutes = st.sidebar.slider('Minutes:', 0, 3000, 360)

#Market Value
st.sidebar.write('Filter players by market value:')
market_value = st.sidebar.slider('Market Value:', 0, 100000000, 7000000)

#Height
st.sidebar.write('Filter players by height:')
height_value = st.sidebar.slider('Height:', 0, 202, 170)


df = df.loc[(df['Minutes played'] > minutes) & (df['Age'] < age) & (df['Position'] != 'GK') & (df['Market value'] < market_value) & (df['Height'] > height_value)]
df.Player.unique()

#-------ASSIGNT VALUES FOR LATER USE------------------

#Assign Player name value for filter
name = df['Player']

# ------------------------------------------------USER INPUT METRICS--------------------


'## CHOOSE METRICS TO CREATE CUSTOM TABLE'
cols = st.multiselect('Metrics:', df.columns, default=['Player', 
                                                       'Team within selected timeframe', 
                                                       'Age', 
                                                       'Minutes played',  
                                                       'Passport country', 
                                                       'xG',  
                                                       'Non-penalty goals', 
                                                       'Goal Ratio', 
                                                       'Shots', 
                                                       'Assists', 
                                                       'xA'])


# show dataframe with the selected columns
st.write(df[cols])


#------------------------------------------------------------ FILTER METRICS FOR PERCENTILE RANKING ---------------------------------------

#Here must be all the values used in Index Rating and Radar
general_midfield_filter = ['Player',
                   'Team', 
                   'Team within selected timeframe', 
                   'Age', 
                   'Height', 
                   'Foot', 
                   'Matches played', 
                   'Contract expires',
                   'Passport country', 
                   'Position', 
                   'Market value', 
                   'Minutes played', 
                   'Assists', 
                   'xA per 90',
                   'Successful attacking actions per 90', 
                   'Key passes per 90',
                   'Shots on target, %', 
                   'Offensive duels won, %', 
                   'Progressive runs per 90', 
                   'Accelerations per 90', 
                   'Successful defensive actions per 90', 
                   'Deep completions per 90', 
                   'Defensive duels won, %', 
                   'Aerial duels won, %', 
                   'nonpenalty_xG/90', 
                   'Defensive duels per 90', 
                   'PAdj Interceptions', 
                   'Accurate forward passes, %', 
                   'Accurate lateral passes, %', 
                   'Accurate short / medium passes, %', 
                   'Accurate long passes, %', 
                   'Accurate passes to final third, %', 
                   'Accurate through passes, %', 
                   'Accurate progressive passes, %', 
                   'Dribbles per 90', 
                   'Successful dribbles, %',
                   'Received passes per 90', 
                   'Assists per 90', 
                   'Non-penalty goals per 90', 
                   'Non-penalty goals', 
                   'xG per 90', 'Shots', 
                   'Shots per 90',
                   'Smart passes per 90',
                   'np_xG_per_shot_average',
                   'Goal Ratio']

#save DF with new column

general_midfield_values = df[general_midfield_filter].copy()




#-------------------------------------------- PERCENTILE TANKING TABS-----------------------


# Values to be used cannot be identical or it will give error saying the multiselect values are identical 


st.title('PERCENTILE RANKING')

general_mid, attacking_mid, defensive_mid, num6_mid = st.tabs(["General Rating", "Attacking Rating", 'Defensive Rating', 'Number 6 Rating'])


##---------------------GENERAL PERCENTILE RANKING----------------------------

with general_mid:


    #user picks which metrics to use for player rating / Metrics that are chosen are turned into percentile ranking

    '#### CHOOSE METRICS TO CREATE PLAYER RATING TABLE 🥇'
    general_rating_filter = st.multiselect('Metrics:', general_midfield_values.columns.difference(['Player', 
                                                                                  'Team', 
                                                                                  'Team within selected timeframe', 
                                                                                  'Shots', 
                                                                                  'Non-penalty goals', 
                                                                                  'Position', 
                                                                                  'Age', 
                                                                                  'Market value', 
                                                                                  'Contract expires', 
                                                                                  'Matches played', 
                                                                                  'Minutes played', 
                                                                                  'Birth country', 
                                                                                  'Passport country', 
                                                                                  'Foot', 
                                                                                  'Height', 
                                                                                  'Weight', 
                                                                                  'On loan', 
                                                                                  'Assists', 
                                                                                  'Non-penalty goals', 
                                                                                  'Shots']), default=['Successful attacking actions per 90', 
                                                                                                      'xA per 90',
                                                                                                      'Key passes per 90', 
                                                                                                      'Shots on target, %', 
                                                                                                      'Offensive duels won, %', 
                                                                                                      'Progressive runs per 90',
                                                                                                      'Successful defensive actions per 90', 
                                                                                                      'Deep completions per 90', 
                                                                                                      'Defensive duels won, %', 
                                                                                                      'nonpenalty_xG/90', 
                                                                                                      'Defensive duels per 90', 
                                                                                                      'PAdj Interceptions', 
                                                                                                      'Accurate forward passes, %', 
                                                                                                      'Accurate passes to final third, %',  
                                                                                                      'Accurate progressive passes, %', 
                                                                                                      'Successful dribbles, %', 
                                                                                                      'Received passes per 90', 
                                                                                                      'Assists per 90', 
                                                                                                      'Non-penalty goals per 90', 
                                                                                                      'Shots per 90',
                                                                                                      'Smart passes per 90',
                                                                                                      'np_xG_per_shot_average',
                                                                                                      'Accurate through passes, %',
                                                                                                      'Goal Ratio'])

    #--------------------------------------------- percentile RANKING INDEX-------------------------------------------

    #Normalize Min/Max Data  ************** Must pass cols as values to normalize  <------------------------------

    scaler = MinMaxScaler()



    general_midfield_values[general_rating_filter] = scaler.fit_transform(general_midfield_values[general_rating_filter]).copy()


    percentile = (general_midfield_values).copy()


    #create index column with average
    percentile['Index'] = general_midfield_values[general_rating_filter].mean(axis=1)

    #normalize index value 0 to 1
    percentile[['Index']] = scaler.fit_transform(percentile[['Index']]).copy()

    #REORDER COLUMNS
    
    #This only effects what gets printed on rating table
    percentile = (percentile[['Player', 
                              'Index', 
                              'Team within selected timeframe', 
                              'Age', 
                              'Contract expires',
                              'Matches played', 
                              'Minutes played', 
                              'Passport country', 
                              'Shots', 
                              'Non-penalty goals', 
                              'xG per 90', 
                              'Non-penalty goals per 90',
                              'xA per 90', 
                              'Assists per 90', 
                              'Shots per 90',  
                              'np_xG_per_shot_average',
                              'Successful attacking actions per 90', 
                              'Key passes per 90',
                              'Smart passes per 90',
                              'Shots on target, %', 
                              'Offensive duels won, %', 
                              'Progressive runs per 90', 
                              'Successful defensive actions per 90', 
                              'Deep completions per 90', 
                              'Defensive duels won, %', 
                              'nonpenalty_xG/90', 
                              'Defensive duels per 90', 
                              'PAdj Interceptions', 
                              'Accurate forward passes, %',  
                              'Accurate passes to final third, %', 
                              'Accurate progressive passes, %', 
                              'Successful dribbles, %',
                              'Received passes per 90', 
                              'Accurate through passes, %',
                              'Goal Ratio']]).copy()

    #Sort By

    percentile = percentile.sort_values('Index', ascending=False).reset_index(drop=True)
    #start index on 1
    percentile.index = percentile.index + 1

    #--------Title

    st.subheader('General Ranking')

    # THIS COLORS THE COLUMNS CHOSEN
    st.write(percentile.style.applymap(styler, subset=['Index', 
                                                       'Successful attacking actions per 90', 
                                                       'Shots on target, %', 
                                                       'np_xG_per_shot_average', 
                                                       'Offensive duels won, %', 
                                                       'Successful defensive actions per 90',
                                                       'Goal Ratio',
                                                       'Defensive duels won, %', 
                                                       'Progressive runs per 90', 
                                                       'Deep completions per 90', 
                                                       'Key passes per 90', 
                                                       'Smart passes per 90',
                                                       'Accurate passes to final third, %', 
                                                       'Accurate through passes, %', 
                                                       'Successful dribbles, %', 
                                                       'Received passes per 90', 
                                                       'nonpenalty_xG/90', 
                                                       'Defensive duels per 90', 
                                                       'PAdj Interceptions', 
                                                       'Accurate forward passes, %', 
                                                       'Accurate progressive passes, %']).set_precision(2))



##---------------------ATTACKING PERCENTILE RANKING----------------------------



with attacking_mid:

    #user picks which metrics to use for player rating / Metrics that are chosen are turned into percentile ranking

    '#### CHOOSE METRICS TO CREATE PLAYER RATING TABLE 🥇'
    attacking_rating_filter = st.multiselect('Metrics:', general_midfield_values.columns.difference(['Player', 
                                                                                  'Team', 
                                                                                  'Team within selected timeframe', 
                                                                                  'Shots', 
                                                                                  'Non-penalty goals', 
                                                                                  'Position', 
                                                                                  'Age', 
                                                                                  'Market value', 
                                                                                  'Contract expires', 
                                                                                  'Matches played', 
                                                                                  'Minutes played', 
                                                                                  'Birth country', 
                                                                                  'Passport country', 
                                                                                  'Foot', 
                                                                                  'Height', 
                                                                                  'Weight', 
                                                                                  'On loan', 
                                                                                  'Assists', 
                                                                                  'Non-penalty goals', 
                                                                                  'Shots']), default=['Successful attacking actions per 90', 
                                                                                                      'xA per 90',
                                                                                                      'Key passes per 90', 
                                                                                                      'Shots on target, %', 
                                                                                                      'Offensive duels won, %', 
                                                                                                      'Progressive runs per 90', 
                                                                                                      'Deep completions per 90',  
                                                                                                      'nonpenalty_xG/90', 
                                                                                                      'Defensive duels per 90', 
                                                                                                      'PAdj Interceptions', 
                                                                                                      'Accurate forward passes, %', 
                                                                                                      'Accurate passes to final third, %',  
                                                                                                      'Accurate progressive passes, %', 
                                                                                                      'Successful dribbles, %', 
                                                                                                      'Received passes per 90', 
                                                                                                      'Assists per 90', 
                                                                                                      'Non-penalty goals per 90', 
                                                                                                      'Shots per 90',
                                                                                                      'Smart passes per 90',
                                                                                                      'np_xG_per_shot_average',
                                                                                                      'Accurate through passes, %',
                                                                                                      'Goal Ratio'])

    #--------------------------------------------- percentile RANKING INDEX-------------------------------------------

    #Normalize Min/Max Data  ************** Must pass cols as values to normalize  <------------------------------

    scaler = MinMaxScaler()



    general_midfield_values[attacking_rating_filter] = scaler.fit_transform(general_midfield_values[attacking_rating_filter]).copy()


    percentile = (general_midfield_values).copy()


    #create index column with average
    percentile['Index'] = general_midfield_values[attacking_rating_filter].mean(axis=1)

    #normalize index value 0 to 1
    percentile[['Index']] = scaler.fit_transform(percentile[['Index']]).copy()

    #REORDER COLUMNS
    
      #This only effects what gets printed on rating table
    percentile = (percentile[['Player', 
                              'Index', 
                              'Team within selected timeframe', 
                              'Age', 
                              'Contract expires',
                              'Height',
                              'Matches played', 
                              'Minutes played', 
                              'Passport country', 
                              'Shots', 
                              'Non-penalty goals', 
                              'xG per 90', 
                              'Non-penalty goals per 90',
                              'xA per 90', 
                              'Assists per 90', 
                              'Shots per 90',  
                              'np_xG_per_shot_average',
                              'Successful attacking actions per 90', 
                              'Key passes per 90',
                              'Smart passes per 90',
                              'Shots on target, %', 
                              'Offensive duels won, %', 
                              'Progressive runs per 90',  
                              'Deep completions per 90',  
                              'nonpenalty_xG/90',  
                              'PAdj Interceptions', 
                              'Accurate forward passes, %',  
                              'Accurate passes to final third, %', 
                              'Accurate progressive passes, %', 
                              'Successful dribbles, %',
                              'Received passes per 90', 
                              'Accurate through passes, %',
                              'Goal Ratio']]).copy()

    #Sort By

    percentile = percentile.sort_values('Index', ascending=False).reset_index(drop=True)
    #start index on 1
    percentile.index = percentile.index + 1

    #--------Title

    st.subheader('Attacking Ranking')

    # THIS COLORS THE COLUMNS CHOSEN
    st.write(percentile.style.applymap(styler, subset=['Index', 
                                                       'Successful attacking actions per 90', 
                                                       'Shots on target, %', 
                                                       'np_xG_per_shot_average', 
                                                       'Offensive duels won, %', 
                                                       'Goal Ratio', 
                                                       'Progressive runs per 90', 
                                                       'Deep completions per 90', 
                                                       'Key passes per 90', 
                                                       'Smart passes per 90',
                                                       'Accurate passes to final third, %', 
                                                       'Accurate through passes, %', 
                                                       'Successful dribbles, %', 
                                                       'Received passes per 90', 
                                                       'nonpenalty_xG/90',  
                                                       'PAdj Interceptions', 
                                                       'Accurate forward passes, %', 
                                                       'Accurate progressive passes, %']).set_precision(2))


##---------------------DEFENSIVE PERCENTILE RANKING----------------------------

with defensive_mid:


    #user picks which metrics to use for player rating / Metrics that are chosen are turned into percentile ranking

    '#### CHOOSE METRICS TO CREATE PLAYER RATING TABLE 🥇'
    general_rating_filter = st.multiselect('Metrics:', general_midfield_values.columns.difference(['Player', 
                                                                                  'Team', 
                                                                                  'Team within selected timeframe', 
                                                                                  'Non-penalty goals', 
                                                                                  'Position', 
                                                                                  'Age', 
                                                                                  'Market value', 
                                                                                  'Contract expires', 
                                                                                  'Matches played', 
                                                                                  'Minutes played', 
                                                                                  'Birth country', 
                                                                                  'Passport country', 
                                                                                  'Foot', 
                                                                                  'Height', 
                                                                                  'Weight', 
                                                                                  'On loan', 
                                                                                  'Assists', 
                                                                                  'Non-penalty goals', 
                                                                                  'Shots']), default=['Successful attacking actions per 90',
                                                                                                      'Successful defensive actions per 90',
                                                                                                      'PAdj Interceptions', 
                                                                                                      'Accurate progressive passes, %'])

    #--------------------------------------------- percentile RANKING INDEX-------------------------------------------

    #Normalize Min/Max Data  ************** Must pass cols as values to normalize  <------------------------------

    scaler = MinMaxScaler()



    general_midfield_values[general_rating_filter] = scaler.fit_transform(general_midfield_values[general_rating_filter]).copy()


    percentile = (general_midfield_values).copy()


    #create index column with average
    percentile['Index'] = general_midfield_values[general_rating_filter].mean(axis=1)

    #normalize index value 0 to 1
    percentile[['Index']] = scaler.fit_transform(percentile[['Index']]).copy()

    #REORDER COLUMNS
    
    #This only effects the final order of the table
    percentile = (percentile[['Player', 
                              'Index', 
                              'Team within selected timeframe', 
                              'Age',
                              'Height',
                              'Contract expires',
                              'Matches played', 
                              'Minutes played', 
                              'Passport country', 
                              'Successful defensive actions per 90',
                              'PAdj Interceptions',
                              'Accurate progressive passes, %']]).copy()

    #Sort By

    percentile = percentile.sort_values('Index', ascending=False).reset_index(drop=True)
    #start index on 1
    percentile.index = percentile.index + 1

    #--------Title

    st.subheader('Defensive Ranking')

    # THIS COLORS THE COLUMNS CHOSEN
    st.write(percentile.style.applymap(styler, subset=['Index',
                                                       'Successful defensive actions per 90', 
                                                       'PAdj Interceptions', 
                                                       'Accurate progressive passes, %']).set_precision(2))
    

    
##---------------------Number 6 PERCENTILE RANKING----------------------------

with num6_mid:


    #user picks which metrics to use for player rating / Metrics that are chosen are turned into percentile ranking

    '#### CHOOSE METRICS TO CREATE PLAYER RATING TABLE 🥇'
    general_rating_filter = st.multiselect('Metrics:', general_midfield_values.columns.difference(['Player', 
                                                                                  'Team', 
                                                                                  'Team within selected timeframe', 
                                                                                  'Non-penalty goals', 
                                                                                  'Position', 
                                                                                  'Age', 
                                                                                  'Market value', 
                                                                                  'Contract expires', 
                                                                                  'Matches played', 
                                                                                  'Minutes played', 
                                                                                  'Birth country', 
                                                                                  'Passport country', 
                                                                                  'Foot', 
                                                                                  'Height', 
                                                                                  'Weight', 
                                                                                  'On loan', 
                                                                                  'Assists', 
                                                                                  'Non-penalty goals', 
                                                                                  'Shots']), default=['Successful attacking actions per 90',
                                                                                                      'Successful defensive actions per 90',
                                                                                                      'Accurate long passes, %',
                                                                                                      'Aerial duels won, %'])

    #--------------------------------------------- percentile RANKING INDEX-------------------------------------------

    #Normalize Min/Max Data  ************** Must pass cols as values to normalize  <------------------------------

    scaler = MinMaxScaler()



    general_midfield_values[general_rating_filter] = scaler.fit_transform(general_midfield_values[general_rating_filter]).copy()


    percentile = (general_midfield_values).copy()


    #create index column with average
    percentile['Index'] = general_midfield_values[general_rating_filter].mean(axis=1)

    #normalize index value 0 to 1
    percentile[['Index']] = scaler.fit_transform(percentile[['Index']]).copy()

    #REORDER COLUMNS
    
    #This only effects the final order of the table
    percentile = (percentile[['Player', 
                              'Index', 
                              'Team within selected timeframe', 
                              'Age',
                              'Height',
                              'Contract expires',
                              'Matches played', 
                              'Minutes played', 
                              'Passport country', 
                              'Successful defensive actions per 90',
                              'Defensive duels won, %',
                              'Accurate long passes, %',
                              'Aerial duels won, %']]).copy()

    #Sort By

    percentile = percentile.sort_values('Index', ascending=False).reset_index(drop=True)
    #start index on 1
    percentile.index = percentile.index + 1

    #--------Title

    st.subheader('Number 6 Ranking')

    # THIS COLORS THE COLUMNS CHOSEN
    st.write(percentile.style.applymap(styler, subset=['Index',
                                                       'Successful defensive actions per 90',
                                                       'Defensive duels won, %',
                                                       'Accurate long passes, %',
                                                       'Aerial duels won, %']).set_precision(2))

#--------------------------------------- TABS ------------------------------

st.title('EFFECTIVENESS METRICS')

tab1, tab2, tab3 = st.tabs(["Shooting", "Dribbling", 'Passing'])


#------------------------------------------------------------------Shooting-------------------------

with tab1:

    st.subheader('SHOOTING SUCCESS RATE')



    #result all 3 aspects *Style Index Colors



    def styler(v):
        if v > 0.08:
            return 'background-color:#E74C3C' #red
        elif v > -0.08:
             return 'background-color:#52CD34' #green
        if v < -0.08:
             return 'background-color:#E74C3C' #red
        # elif v < .40:
        #     return 'background-color:#E67E22' #orange
        # else:
        #     return 'background-color:#F7DC6F'  #yellow


    #Sort By

    shooting = df.sort_values('Shots', ascending=False)


    #Choose columns to show

    shooting = (shooting[['Player', 
              'Team', 
              'Minutes played', 
              'Shots',
              'np_xG_per_shot_average',
              'Goal Ratio',
              'xG_Difference',
              'Non-penalty goals',
              'nonpenalty_xG/90', 
              'Non-penalty goals per 90',     
              'nonpenalty_xG', 
              'Position', 
              'Passport country', 
              'Age', 
              '90s', 
              'Shots per 90']])


    # print table

    st.write(shooting.style.applymap(styler, subset=['xG_Difference']).set_precision(2))
    

#------------------------------------------------------------------Passing------------------------- 

with tab3:
    
    st.subheader('PASSING SUCCESS')
    smartpassing = df.sort_values('Accurate smart passes, %', ascending=False)

    #dribble success flter
    st.write('Filter players by Smart passes per 90m:')
    smartpassesx90 = st.slider('Smart passes per 90:',  0.0, 5.0, 0.5)


    smartpassing = smartpassing[~(smartpassing['Smart passes per 90'] <= smartpassesx90)] 
    smartpassing.index = range(len(smartpassing.index))
    smartpassing = smartpassing.round(2)

    #No decimals
    #smartpassing['Accurate smart passes, %'] = smartpassing['Accurate smart passes, %'].astype(str).apply(lambda x: x.replace('.0',''))

    #Add % sign
    smartpassing['Accurate smart passes, %'] = smartpassing['Accurate smart passes, %'].astype(str) + '%'


    #rename 


    smartpassing.rename(columns={'Accurate smart passes, %':'% of Accurate smart passes'}, inplace=True)


    smartpassing = smartpassing.reset_index(drop=True)
    smartpassing.index = smartpassing.index + 1

    pd.set_option('display.max_rows', smartpassing.shape[0]+1)
    st.write((smartpassing[['Player','Smart passes per 90', '% of Accurate smart passes', 'Key passes per 90', 'Team', 
                            'Age', 'Passport country', 'Market value', 'Contract expires']]))



#------------------------------------------------------------------Dribble------------------------- 

with tab2:
    
    st.subheader('DRIBBLE SUCCESS RATE')
    dribbling = df.sort_values('Successful dribbles, %', ascending=False)

    #dribble success flter
    st.write('Filter players by dribbles per 90m:')
    driblesx90 = st.slider('Dribbles per 90m:',  0, 7, 3)


    dribbling = dribbling[~(dribbling['Dribbles per 90'] <= driblesx90)] 
    dribbling.index = range(len(dribbling.index))
    dribbling = dribbling.round()

    #No decimals
    dribbling['Successful dribbles, %'] = dribbling['Successful dribbles, %'].astype(str).apply(lambda x: x.replace('.0',''))

    #Add % sign
    dribbling['Successful dribbles, %'] = dribbling['Successful dribbles, %'].astype(str) + '%'


    #rename 


    dribbling.rename(columns={'Successful dribbles, %':'% of Successful dribbles'}, inplace=True)


    dribbling = dribbling.reset_index(drop=True)
    dribbling.index = dribbling.index + 1

    pd.set_option('display.max_rows', dribbling.shape[0]+1)
    st.write((dribbling[['Player','Dribbles per 90', '% of Successful dribbles', 'Team', 'Age', 'Passport country', 'Market value', 'Contract expires']]))
    
    
#-------------------------------------------------------------------Predict Value----------------------------------------

st.title('PREDICT PLAYER VALUES 🔮💰')

### Define values to use

predict_df = df[['Player',
         'Team',
         'Position',
         'Age',
         'Market value',
         'Minutes played',
         'Height',
         'Weight',
         'xG per 90',
         'Head goals',
         'Shots on target, %',
         'Assists per 90',
         'Crosses to goalie box per 90',
         'Dribbles per 90',
         'Successful dribbles, %',
         'Progressive runs per 90',
         'Accelerations per 90',
         'Received passes per 90',
         'Fouls suffered per 90',
         'Back passes per 90',
         'Second assists per 90',
         'Third assists per 90',
         'Smart passes per 90',
         'Key passes per 90',
         'Passes to final third per 90',
         'Accurate passes to penalty area, %',
         'Through passes per 90',
         'Accurate through passes, %',
         'Deep completions per 90',
         'Corners per 90',
         'Penalties taken']]

#assign values to add back into df
#select unique names to avoid conflicts with radar
value_predict = predict_df['Market value']
name_predict = predict_df['Player']
team_predict = predict_df['Team']
minutes_predict = predict_df['Minutes played']
position_predict = predict_df['Position']

predict_df = predict_df.drop(['Market value', 'Player', 'Team', 'Minutes played', 'Position'], axis=1)

#add market value back into end of df

predict_df['value'] = value_predict



### Create DF NP to create x and y train df's

df_np = predict_df.to_numpy()

#take the first N columns and assign them to x train and the last column (MARKET VALUE) and assign it to y train

X_train, y_train = df_np[:, :26], df_np[:, -1]

from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression().fit(X_train, y_train)
sklearn_y_predictions = sklearn_model.predict(X_train).astype(int)

predictions_df = pd.DataFrame({
                               'xG per 90': predict_df['xG per 90'],
                               'Assists per 90': predict_df[ 'Assists per 90'],
                               'Key passes per 90': predict_df['Key passes per 90'],
                               'Age': predict_df['Age'],
                               'Value': predict_df['value'],
                               'Value Prediction':sklearn_y_predictions})

#add back name and team columns
predictions_df['Name'] = name_predict
predictions_df['Minutes played'] = minutes_predict
predictions_df['Team'] = team_predict

#reorder columns
predictions_df = predictions_df[['Name', 'Team', 'Age', 'Value', 'Value Prediction', 'Minutes played']]
predictions_df.reset_index(drop=True, inplace=True)

#format using commas for value
#predictions_df['Value'] = predictions_df.apply(lambda x: "{:,}".format(x['Value']), axis=1)
#predictions_df['Value Prediction'] = predictions_df.apply(lambda x: "{:,}".format(x['Value Prediction']), axis=1)

st.write(predictions_df)


#------------------------------------------------------------------------------------------RADAR----------------------------------------

st.title('PERCENTILE RANKING RADAR')


#Define colors
    
Background = '#020807'
Defense = '#E67E22'
Passes = '#3f88c5'
Attack = '#E1005C'
Black = '#fffcf2' #Inner text color 
TempColor = '#FFD15C'
LeagueColor = '#FF3333'
PosColor = '#72A8D5'


#Title 
title_font = FontManager(("https://github.com/Mozi333/fonts/blob/main/RecoletaAlt-Black.ttf?raw=true"))

#Sutitle
subtitle_font = FontManager(("https://github.com/Mozi333/fonts/blob/main/RightGrotesk-Medium.otf?raw=true"))

#Values Text
font_text = FontManager(("https://github.com/google/fonts/blob/main/ofl/lato/Lato-SemiBold.ttf?raw=true"))
bio_text = FontManager(("https://github.com/google/fonts/blob/main/ofl/lato/Lato-SemiBold.ttf?raw=true"))


#------ Player name filter

#name has a list with all the player names
option = st.selectbox('Choose Player', (name))

st.write('You chose:', option)

fullname = st.text_input('Type name to print on radar', option)
st.write('The full name of the player is:', fullname)


#---------------- PRINT RADAR -----------------------------------
def radar(general_midfield_values, name, minutes, age, SizePlayer):
    
        
    #-----Define Bio values
    
    #Define Team
    
    Team = general_midfield_values[general_midfield_values['Player']==option]
    Team = Team['Team within selected timeframe'].item()
    
    #Define Age
    
    Age = general_midfield_values[general_midfield_values['Player']==option]
    Age = Age['Age'].item()
    
    #Define Height
    
    Height = general_midfield_values[general_midfield_values['Player']==option]
    Height = Height['Height'].item()
    
    #Define Foot
    
    Foot = general_midfield_values[general_midfield_values['Player']==option]
    Foot = Foot['Foot'].item()
    

    #Define Matches Played
    
    Matches = general_midfield_values[general_midfield_values['Player']==option]
    Matches = Matches['Matches played'].item()
    
    #Define Goals
    
    Goals = general_midfield_values[general_midfield_values['Player']==option]
    Goals = Goals['Non-penalty goals'].item()
    
    #Define Assists
    
    Assists = general_midfield_values[general_midfield_values['Player']==option]
    Assists = Assists['Assists'].item()
    
    #Define Minutes Played
    
    Minutesplayed = general_midfield_values[general_midfield_values['Player']==option]
    Minutesplayed = Minutesplayed['Minutes played'].item()

    
    #Define Market Value
    
    Marketvalue = general_midfield_values[general_midfield_values['Player']==option]
    Marketvalue = Marketvalue['Market value'].item()
    
    #Define Contract Info
    
    Contractexpires = general_midfield_values[general_midfield_values['Player']==option]
    Contractexpires = Contractexpires['Contract expires'].item()

    #Rename Values


    general_midfield_values.rename(columns={
        'xA per 90':'xA p90m',
        'Successful defensive actions per 90':'Successful \ndefensive \nactions \np90m',
        'Successful attacking actions per 90':'Successful \nattacking \nactions \np90m',
        'Accelerations per 90':'Accelerations \np90m',
        'Key passes per 90':'Key \npasses \np90m',
        'Shots on target, %':'% Shots \non target',
        'Progressive runs per 90':'Progressive \nruns p90m',
        'Offensive duels won, %':'% Offensive \nduels won',
        'Goal Ratio':'Goal \nRatio',
        'nonpenalty_xG/90':'xG \np90m',
        'Accurate passes to final third, %':'% Accurate \npasses to \nfinal third',
        'Non-penalty goals per 90':'Goals \np90m',
        'Shots per 90':'Shots \np90m',
        'Accurate through passes, %':'% Accurate \nthrough \npasses',
        'PAdj Interceptions':'PAdj \nInterceptions',
        'Received passes per 90':'Received \npasses \np90m',
        'Defensive duels won, %':'% Defensive \nduels \nwon',
        'np_xG_per_shot_average':'xG per \nshot \navg',
        'Aerial duels won, %':'% Aerial \nduels \nwon'}, inplace=True)

    #REORDER COLUMNS
    
    #This only effects the final order of the table

    general_midfield_values = general_midfield_values[[
            'Player',
            'xG \np90m',
            'Goals \np90m',
            'xA p90m',
            'Goal \nRatio',
            'Shots \np90m',
            'xG per \nshot \navg',
            '% Offensive \nduels won',
            'Successful \nattacking \nactions \np90m',
            'Progressive \nruns p90m',
            'Accelerations \np90m',
            '% Shots \non target',
            'Key \npasses \np90m',
            '% Accurate \npasses to \nfinal third',
            '% Accurate \nthrough \npasses',
            'Received \npasses \np90m',
            'Successful \ndefensive \nactions \np90m',
            '% Defensive \nduels \nwon',
            'PAdj \nInterceptions',
            '% Aerial \nduels \nwon']]
    
    #Create a parameter list
    
    params = list(general_midfield_values.columns)
    
    #drop player column
    
    params = params[1:]
    
    # Now we filter the df for the player we want.
    # The player needs to be spelled exactly the same way as it is in the data. Accents and everything 
    
    player = general_midfield_values.loc[general_midfield_values['Player']==option].reset_index()
    player = list(player.loc[0]) #gets all values/rows of specific player
    player = player[2:]
    
    # now that we have the player scores, we need to calculate the percentile values with scipy stats.
    # I am doing this because I do not know the percentile beforehand and only have the raw numbers
    
    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(general_midfield_values[params[x]],player[x])))
        
    
    #------Plot Radar

    # color for the slices and text
    slice_colors = [Attack] * 11 + [Passes] * 4 + [Defense] * 4  # ataque - pases 
    text_colors = ["#F2F2F2"] * 19

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        background_color=Background,     # background color
        straight_line_color="#FFFFFF",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=0,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=9            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(6,4),                   # adjust figsize according to your need
        param_location=115,              # where the parameter names will be added adjust position
        color_blank_space="same",        # use same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,        # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="#FFFFFF", zorder=2, linewidth=2
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="#FFFFFF", fontsize=7, #size of value titles
            fontproperties=font_text.prop, va="center"
        ),                               # values to be used when adding parameter
        kwargs_values=dict(
            color="#000000", fontsize=7, #size of values numbers
            fontproperties=font_text.prop, zorder=3,
            bbox=dict(
                edgecolor="#FFFFFF", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )                                # values to be used when adding parameter-values
    )
    

    # add name title
    
    fig_text(0, 1.18, f'<{str(fullname.upper(),)}>', size = SizePlayer,  fontproperties=title_font.prop, color=Black, 
        highlight_textprops=[{'color':Black}])

    
    # add credits
    notes = '* Only players with more than ' + str(minutes) + ' minutes played' + '\n* Only players under ' + str(age) + '\n* Penalty goals are not counted' + '\n Data from: Wyscout'


    fig.text(
        1.1, -0.09, f"{notes}", size=7, #does not include player text
        fontproperties=font_text.prop, color=Black,
        ha="right"
    )
    
    # add rectangle text

    fig.text(
        0.78, .051, "Passing", size=6,     #passing
        fontproperties=bio_text.prop, color=Black
    )
    
    # add text
    fig.text(
        0.86, 0.051, "Defense", size=6,   #defense
        fontproperties=bio_text.prop, color=Black
    )
    
    # add text
    fig.text(
        0.95, 0.051, "Attack", size=6,    #attack
        fontproperties=bio_text.prop, color=Black
    )
    
    

    
    # add rectangles
    fig.patches.extend([
        
        plt.Rectangle(
            (0.75, 0.05), 0.025, 0.021, fill=True, color=Passes, #passing
            transform=fig.transFigure, figure=fig
        ),
        
        plt.Rectangle(
            (0.83, 0.05), 0.025, 0.021, fill=True, color=Defense, #defense
            transform=fig.transFigure, figure=fig
        ),
        
        plt.Rectangle(
            (0.92, 0.05), 0.025, 0.021, fill=True, color=Attack, #attack
            transform=fig.transFigure, figure=fig
        ),
        

    ])
    
    #BIO----------

    
    #team
    
    fig.text(
        
        
        0.90, 0.40,  str(Team), size=10,    #team
        fontproperties=bio_text.prop, color=Black
    )
    
    #Age
    
    fig.text(
        
        
        0.90, 0.35,  "Age: " + str(Age), size=7,    #Edad
        fontproperties=bio_text.prop, color=Black
    )
    
    #Altura
    
    fig.text(
        
        
        0.90, 0.32,  "Height: " + str(Height), size=7,    #Height
        fontproperties=bio_text.prop, color=Black
    )
    
    #Foot
    
    fig.text(
        
        
        0.90, 0.29,  "Foot: " + str(Foot), size=7,    #Foot
        fontproperties=bio_text.prop, color=Black
    )
    
    
    
    #Matches Played
    
    fig.text(
        
        
        0.90, 0.26,  "Matches: " + str(Matches), size=7,    
        fontproperties=bio_text.prop, color=Black
    )
    
    #Minutes played
    
    fig.text(
        
        
        0.90, 0.23,  "Minutes played: " + str(Minutesplayed), size=7,    
        fontproperties=bio_text.prop, color=Black
    )
        
    #Goals
    
    fig.text(
        
        
        0.90, 0.20,  "Goals: " + str(Goals), size=7,    
        fontproperties=bio_text.prop, color=Black
    )
    
    #Assists
    
    fig.text(
        
        
        0.90, 0.17,  "Assists: " + str(Assists), size=7,    
        fontproperties=bio_text.prop, color=Black
    )
    
    
    #Market value
    
    fig.text(
        
        
        0.90, 0.14,  "Market value:  €" + str(f"{Marketvalue:,}"), size=7,    
        fontproperties=bio_text.prop, color=Black
    )
    
    #Contract info
    
    fig.text(
        
        
        0.90, 0.11,  "Contract exp: " + str(Contractexpires), size=7,    
        fontproperties=bio_text.prop, color=Black
    )

    st.pyplot(fig)
   

radar(general_midfield_values, option, minutes, age, SizePlayer = 45)

