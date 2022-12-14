import streamlit as st
st.set_page_config(layout="wide",
                  page_title="Defender Scout",
                  page_icon='💣')
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
#import new metrics module from new_metrics.py folder
from new_metrics import *


#-----------------------------FUNCTIONS--------------------------------------------------------------

st.title('💣 DEFENCE SCOUT')


def load_data():
    
    data = (r'https://github.com/Mozi333/X/blob/main/data/all-CB-b-europe-2022.xlsx?raw=true')
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


    
#Divide players per position

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


st.sidebar.title('Set Filters')

#age
st.sidebar.write('Filter players by age:')
age = st.sidebar.slider('Age:', 0, 45, 40)


#Minutes Played
st.sidebar.write('Filter players by minutes played:')
minutes = st.sidebar.slider('Minutes:', 0, 3000, 380)


#Market Value
st.sidebar.write('Filter players by market value:')
market_value = st.sidebar.slider('Market Value:', 0, 100000000, 7000000)

#Height
st.sidebar.write('Filter players by height:')
height_value = st.sidebar.slider('Height:', 0, 202, 0)

#dribbles
st.sidebar.write('Filter players by dribbling:')
Dribbles_per_90 = st.sidebar.slider('Dribbles per 90:', 0.0, 10.0, 0.0)
Successful_dribbles_pct = st.sidebar.slider('% Successful dribbles:', 0.0, 100.0, 0.0)


df = df.loc[(df['Minutes played'] > minutes) & (df['Age'] < age) & (df['Position'] != 'GK') & (df['Market value'] < market_value) & (df['Height'] > height_value) 
            & (df['Dribbles per 90'] > Dribbles_per_90) & (df['Successful dribbles, %'] > Successful_dribbles_pct)]

df.Player.unique() 


#-------ASSIGNT VALUES FOR LATER USE------------------

#Assign Player name value for filter
name = df['Player']

# # ------------------------------------------------USER INPUT METRICS--------------------


# '## CHOOSE METRICS TO CREATE CUSTOM TABLE'
# cols = st.multiselect('Metrics:', df.columns, default=['Player', 
#                                                        'Team within selected timeframe', 
#                                                        'Age', 
#                                                        'Minutes played',  
#                                                        'Passport country', 
#                                                        'xG', 
#                                                        'Defensive duels per 90', 
#                                                        'Defensive duels won, %', 
#                                                        'Aerial duels per 90', 
#                                                        'Aerial duels won, %', 
#                                                        'Sliding tackles per 90', 
#                                                        'PAdj Sliding tackles', 
#                                                        'Shots blocked per 90',
#                                                        'Interceptions per 90',
#                                                        'PAdj Interceptions',
#                                                        'Fouls per 90',
#                                                        'Yellow cards per 90',
#                                                        'Red cards per 90'])


# # show dataframe with the selected columns
# st.write(df[cols])

#------------------------------------------------------------ FILTER METRICS ---------------------------------------

#Here must be all the values used in Index Rating and Radar
defender_filter = ['Player',
                 'Team',
                 'Team within selected timeframe',
                 'Position',
                 'Age',
                 'Market value',
                 'Contract expires',
                 'Matches played',
                 'Minutes played',
                 'Goals',
                 'xG',
                 'Assists',
                 'xA',
                 'Duels per 90',
                 'Duels won, %',
                 'Birth country',
                 'Passport country',
                 'Foot',
                 'Height',
                 'Weight',
                 'On loan',
                 'Successful defensive actions per 90',
                 'Defensive duels per 90',
                 'Defensive duels won, %',
                 'Aerial duels per 90',
                 'Aerial duels won, %',
                 'Sliding tackles per 90',
                 'PAdj Sliding tackles',
                 'Shots blocked per 90',
                 'Interceptions per 90',
                 'PAdj Interceptions',
                 'Fouls per 90',
                 'Yellow cards',
                 'Yellow cards per 90',
                 'Red cards',
                 'Red cards per 90',
                 'Successful attacking actions per 90',
                 'Goals per 90',
                 'Non-penalty goals',
                 'Non-penalty goals per 90',
                 'xG per 90',
                 'Head goals',
                 'Head goals per 90',
                 'Shots',
                 'Shots per 90',
                 'Shots on target, %',
                 'Goal conversion, %',
                 'Assists per 90',
                 'Crosses per 90',
                 'Accurate crosses, %',
                 'Crosses from left flank per 90',
                 'Accurate crosses from left flank, %',
                 'Crosses from right flank per 90',
                 'Accurate crosses from right flank, %',
                 'Crosses to goalie box per 90',
                 'Dribbles per 90',
                 'Successful dribbles, %',
                 'Offensive duels per 90',
                 'Offensive duels won, %',
                 'Touches in box per 90',
                 'Progressive runs per 90',
                 'Accelerations per 90',
                 'Received passes per 90',
                 'Received long passes per 90',
                 'Fouls suffered per 90',
                 'Passes per 90',
                 'Accurate passes, %',
                 'Forward passes per 90',
                 'Accurate forward passes, %',
                 'Back passes per 90',
                 'Accurate back passes, %',
                 'Lateral passes per 90',
                 'Accurate lateral passes, %',
                 'Short / medium passes per 90',
                 'Accurate short / medium passes, %',
                 'Long passes per 90',
                 'Accurate long passes, %',
                 'Average pass length, m',
                 'Average long pass length, m',
                 'xA per 90',
                 'Shot assists per 90',
                 'Second assists per 90',
                 'Third assists per 90',
                 'Smart passes per 90',
                 'Accurate smart passes, %',
                 'Key passes per 90',
                 'Passes to final third per 90',
                 'Accurate passes to final third, %',
                 'Passes to penalty area per 90',
                 'Accurate passes to penalty area, %',
                 'Through passes per 90',
                 'Accurate through passes, %',
                 'Deep completions per 90',
                 'Deep completed crosses per 90',
                 'Progressive passes per 90',
                 'Accurate progressive passes, %',
                 'Aerial duels per 90.1',
                 'Free kicks per 90',
                 'Direct free kicks per 90',
                 'Direct free kicks on target, %',
                 'Corners per 90',
                 'Penalties taken',
                 'Penalty conversion, %',
                 'Goal %',
                 'Goal Ratio',
                 '90s',
                 'penalty_xG',
                 'nonpenalty_xG',
                 'penalty_xG/90',
                 'nonpenalty_xG/90',
                 'Sum_xGp90_and_Goalsx90',
                 'Sum_xAx90_and_Assistx90',
                 'xG_Difference',
                 'Main_position',
                 'Field_position',
                 'Duels  Total',
                 'Successful defensive actions  Total',
                 'Defensive duels  Total',
                 'Aerial duels  Total',
                 'Sliding tackles  Total',
                 'Shots blocked  Total',
                 'Interceptions  Total',
                 'Fouls  Total',
                 'Yellow cards  Total',
                 'Red cards  Total',
                 'Successful attacking actions  Total',
                 'Goals  Total',
                 'Non-penalty goals  Total',
                 'xG  Total',
                 'Head goals  Total',
                 'Shots  Total',
                 'Assists  Total',
                 'Crosses  Total',
                 'Crosses from left flank  Total',
                 'Crosses from right flank  Total',
                 'Crosses to goalie box  Total',
                 'Dribbles  Total',
                 'Offensive duels  Total',
                 'Touches in box  Total',
                 'Progressive runs  Total',
                 'Accelerations  Total',
                 'Received passes  Total',
                 'Received long passes  Total',
                 'Fouls suffered  Total',
                 'Passes  Total',
                 'Forward passes  Total',
                 'Back passes  Total',
                 'Lateral passes  Total',
                 'Short / medium passes  Total',
                 'Long passes  Total',
                 'xA  Total',
                 'Shot assists  Total',
                 'Second assists  Total',
                 'Third assists  Total',
                 'Smart passes  Total',
                 'Key passes  Total',
                 'Passes to final third  Total',
                 'Passes to penalty area  Total',
                 'Through passes  Total',
                 'Deep completions  Total',
                 'Deep completed crosses  Total',
                 'Progressive passes  Total',
                 'Free kicks  Total',
                 'Direct free kicks  Total',
                 'Corners  Total',
                 'np_xG_per_shot_average']

#save DF with Striker filter columns

defender_values = df[defender_filter].copy()

#user picks which metrics to use for player rating

'## CHOOSE METRICS TO CREATE PLAYER RATING TABLE 🥇'
ratingfilter = st.multiselect('Metrics:', defender_values.columns.difference(['Player', 
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
                                                                            'Main_position']), 
                              default=['Defensive duels won, %', 
                                       'Aerial duels won, %',
                                       'PAdj Interceptions',
                                       'Accurate lateral passes, %',
                                       'Successful defensive actions per 90',
                                       'Dribbles per 90',
                                       'Accurate passes, %',
                                       'Accelerations per 90',
                                       'PAdj Interceptions',
                                       'Accurate progressive passes, %'])


#--------------------------------------------- percentile RANKING INDEX-------------------------------------------

#Normalize Min/Max Data  ************** Must pass cols as values to normalize  <------------------------------

scaler = MinMaxScaler()


#use indexfilter metrics to create player INDEX
defender_values[ratingfilter] = scaler.fit_transform(defender_values[ratingfilter]).copy()


percentile = (defender_values).copy()


#create index column with average
percentile['Index'] = defender_values[ratingfilter].mean(axis=1)

#turn index into 0-1 percentile
percentile[['Index']] = scaler.fit_transform(percentile[['Index']]).copy()

#Make filters based on percentile ranking
st.sidebar.write('Filter players by metrics percentile:')
Defensive_Duels_Won = st.sidebar.slider('Defensice Duels:', 0.0, 1.0, 0.0)
Accurate_progressive_passes = st.sidebar.slider('Accurate progressive passes:', 0.0, 1.0, 0.0)
PAdj_Interceptions = st.sidebar.slider('PAdj Interceptions:', 0.0, 1.0, 0.0)
Smart_passes_per_90 = st.sidebar.slider('Smart passes per 90:', 0.0, 1.0, 0.0)


percentile = percentile.loc[(percentile['Defensive duels won, %'] > Defensive_Duels_Won) & (percentile['Accurate progressive passes, %'] > Accurate_progressive_passes) & (percentile['PAdj Interceptions'] > PAdj_Interceptions) & (percentile['Smart passes per 90'] > Smart_passes_per_90)] 


#reorder columns
#This marks what columns are shown in rating table
percentile = (percentile[['Player', 
                          'Index', 
                          'Team within selected timeframe', 
                          'Age', 
                          'Height',
                          'Weight',
                          'Contract expires',
                          'Market value',
                          'Position', 
                          'Matches played', 
                          'Minutes played', 
                          'Passport country', 
                          'Defensive duels won, %', 
                          'Aerial duels won, %',
                          'PAdj Interceptions',
                          'Accurate lateral passes, %',
                          'Successful defensive actions per 90',
                          'Dribbles per 90',
                          'Accurate passes, %',
                          'Accelerations per 90',
                          'Accurate progressive passes, %',
                          'Smart passes per 90']]).copy()

#Sort By

percentile = percentile.sort_values('Index', ascending=False).reset_index(drop=True)
#start index on 1
percentile.index = percentile.index + 1

#--------Title

st.title('PERCENTILE RANKING')

# THIS COLORS THE COLUMNS CHOSEN
st.write(percentile.style.applymap(styler, subset=['Index',
                                                   'Defensive duels won, %',
                                                   'Successful defensive actions per 90',
                                                   'Aerial duels won, %',
                                                   'PAdj Interceptions',
                                                   'Accurate lateral passes, %',
                                                   'Dribbles per 90',
                                                   'Accurate passes, %',
                                                   'Accelerations per 90',
                                                   'Accurate progressive passes, %',
                                                   'Smart passes per 90']).set_precision(2))


# #--------------------------------------- TABS ------------------------------

# st.title('EFFECTIVENESS METRICS')

# tab1, tab2 = st.tabs(["Shooting", "Dribbling"])


# #------------------------------------------------------------------Shooting-------------------------

# with tab1:

#     st.subheader('SHOOTING SUCCESS RATE')



#     #result all 3 aspects *Style Index Colors



#     def styler(v):
#         if v > 0.08:
#             return 'background-color:#E74C3C' #red
#         elif v > -0.08:
#              return 'background-color:#52CD34' #green
#         if v < -0.08:
#              return 'background-color:#E74C3C' #red
#         # elif v < .40:
#         #     return 'background-color:#E67E22' #orange
#         # else:
#         #     return 'background-color:#F7DC6F'  #yellow


#     #Sort By

#     shooting = df.sort_values('Shots', ascending=False)


#     #Choose columns to show

#     shooting = (shooting[['Player', 
#               'Team', 
#               'Minutes played', 
#               'Shots',
#               'Goal Ratio',
#               'xG_Difference',
#               'Non-penalty goals',
#               'nonpenalty_xG/90', 
#               'Non-penalty goals per 90',     
#               'nonpenalty_xG', 
#               'Position', 
#               'Passport country', 
#               'Age', 
#               '90s', 
#               'Shots per 90']])


#     # print table

#     st.write(shooting.style.applymap(styler, subset=['xG_Difference']).set_precision(2))



# #------------------------------------------------------------------Dribble------------------------- 

# with tab2:
    
#     st.subheader('DRIBBLE SUCCESS RATE')
#     dribbling = df.sort_values('Successful dribbles, %', ascending=False)

#     #dribble success flter
#     st.write('Filter players by dribbles per 90m:')
#     driblesx90 = st.slider('Dribbles per 90m:',  0, 7, 3)


#     dribbling = dribbling[~(dribbling['Dribbles per 90'] <= driblesx90)] 
#     dribbling.index = range(len(dribbling.index))
#     dribbling = dribbling.round()

#     #No decimals
#     dribbling['Successful dribbles, %'] = dribbling['Successful dribbles, %'].astype(str).apply(lambda x: x.replace('.0',''))

#     #Add % sign
#     dribbling['Successful dribbles, %'] = dribbling['Successful dribbles, %'].astype(str) + '%'


#     #rename 


#     dribbling.rename(columns={'Successful dribbles, %':'% of Successful dribbles'}, inplace=True)


#     dribbling = dribbling.reset_index(drop=True)
#     dribbling.index = dribbling.index + 1

#     pd.set_option('display.max_rows', dribbling.shape[0]+1)
#     st.write((dribbling[['Player','Dribbles per 90', '% of Successful dribbles', 'Team', 'Age', 'Passport country', 'Market value', 'Contract expires']]))


#---------------------------------------------------------------------RADAR----------------------------------------

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
st.write('The full name of the player is:', option)


#---------------- PRINT RADAR -----------------------------------
def radar(defender_values, name, minutes, age, SizePlayer):
    
        
    #-----Define Bio values
    
    #Define Team
    
    Team = defender_values[defender_values['Player']==option]
    Team = Team['Team within selected timeframe'].item()
    
    #Define Age
    
    Age = defender_values[defender_values['Player']==option]
    Age = Age['Age'].item()
    
    #Define Height
    
    Height = defender_values[defender_values['Player']==option]
    Height = Height['Height'].item()
    
    #Define Foot
    
    Foot = defender_values[defender_values['Player']==option]
    Foot = Foot['Foot'].item()
    

    #Define Matches Played
    
    Matches = defender_values[defender_values['Player']==option]
    Matches = Matches['Matches played'].item()
    
    #Define Goals
    
    Goals = defender_values[defender_values['Player']==option]
    Goals = Goals['Non-penalty goals'].item()
    
    #Define Assists
    
    Assists = defender_values[defender_values['Player']==option]
    Assists = Assists['Assists'].item()
    
    #Define Minutes Played
    
    Minutesplayed = defender_values[defender_values['Player']==option]
    Minutesplayed = Minutesplayed['Minutes played'].item()

    
    #Define Market Value
    
    Marketvalue = defender_values[defender_values['Player']==option]
    Marketvalue = Marketvalue['Market value'].item()
    
    #Define Contract Info
    
    Contractexpires = defender_values[defender_values['Player']==option]
    Contractexpires = Contractexpires['Contract expires'].item()

    #Rename Values


    defender_values.rename(columns={
        'Defensive duels won, %':'% Defensive \nduels \nwon', 
       'Aerial duels won, %':'% Aerial \nduels \nwon', 
       'PAdj Sliding tackles':'PAdj \nSliding \ntackles', 
       'Shots blocked per 90':'Shots \nblocked \np90m',
       'PAdj Interceptions':'PAdj \nInterceptions',
        'Accurate forward passes, %':'% Accurate \nforward \npasses',
        'Accurate lateral passes, %':'% Accurate \nlateral \npasses',
        'Accurate long passes, %':'% Accurate \nlong \npasses',
        'Smart passes per 90':'Smart \npasses \np90m',
        'Accurate progressive passes, %':'% Accurate \nprogressive \npasses'}, inplace=True)


    #Reorder Values

    defender_values = defender_values[[
            'Player',
            '% Defensive \nduels \nwon',
            '% Aerial \nduels \nwon',
            'PAdj \nSliding \ntackles',
            'Shots \nblocked \np90m',
            'PAdj \nInterceptions',
            '% Accurate \nforward \npasses',
            '% Accurate \nlateral \npasses',
            '% Accurate \nlong \npasses',
            'Smart \npasses \np90m',
            '% Accurate \nprogressive \npasses']]
    
    #Create a parameter list
    
    params = list(defender_values.columns)
    
    #drop player column
    
    params = params[1:]
    
    # Now we filter the df for the player we want.
    # The player needs to be spelled exactly the same way as it is in the data. Accents and everything 
    
    player = defender_values.loc[defender_values['Player']==option].reset_index()
    player = list(player.loc[0]) #gets all values/rows of specific player
    player = player[2:]
    
    # now that we have the player scores, we need to calculate the percentile values with scipy stats.
    # I am doing this because I do not know the percentile beforehand and only have the raw numbers
    
    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(defender_values[params[x]],player[x])))
        
    
    #------Plot Radar


    # color for the slices and text
    slice_colors = [Defense] * 5 + [Passes] * 5 # defensa - pases
    text_colors = ["#F2F2F2"] * 10

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


radar(defender_values, option, minutes, age, SizePlayer = 45)

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
 'Defensive duels won, %',
 'Aerial duels won, %',
 'PAdj Sliding tackles',
 'Shots blocked per 90',
 'PAdj Interceptions',
 'Fouls per 90',
 'Yellow cards per 90',
 'Red cards per 90',
 'xG per 90',
 'Head goals per 90',
 'Shots per 90',
 'Shots on target, %',
 'Goal conversion, %',
 'Assists per 90',
 'Accurate crosses, %',
 'Accurate crosses from left flank, %',
 'Accurate crosses from right flank, %',
 'Crosses to goalie box per 90',
 'Dribbles per 90',
 'Successful dribbles, %',
 'Offensive duels won, %',
 'Touches in box per 90',
 'Progressive runs per 90',
 'Accelerations per 90',
 'Received long passes per 90',
 'Fouls suffered per 90',
 'Accurate forward passes, %',
 'Back passes per 90',
 'Accurate back passes, %',
 'Lateral passes per 90',
 'Accurate lateral passes, %',
 'Long passes per 90',
 'Accurate long passes, %',
 'Average pass length, m',
 'Average long pass length, m',
 'xA per 90',
 'Second assists per 90',
 'Third assists per 90',
 'Smart passes per 90',
 'Accurate smart passes, %',
 'Key passes per 90',
 'Accurate passes to final third, %',
 'Accurate passes to penalty area, %',
 'Through passes per 90',
 'Accurate through passes, %',
 'Deep completions per 90',
 'Deep completed crosses per 90',
 'Progressive passes per 90',
 'Accurate progressive passes, %',
 'Free kicks per 90',
 'Direct free kicks per 90',
 'Direct free kicks on target, %',
 'Corners per 90',
 'Penalties taken',
 'Penalty conversion, %']]

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

X_train, y_train = df_np[:, :56], df_np[:, -1]

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
