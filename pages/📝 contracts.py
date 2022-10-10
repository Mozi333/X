# Base
import pandas as pd
import numpy as np

# Visualización
import plotly.express as ex
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Web Scraping
import requests
from bs4 import BeautifulSoup
import pyjsparser

# trabajo con variables abocadas al tiempo
import sys
import time
from datetime import datetime
from termcolor import colored

# GC
import gc

# Itertools
import itertools

# Para extraer datos de un gráfico
import js2xml
from itertools import repeat    
from pprint import pprint as pp

# Expresiones regulares
import re 

# Configuración
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

#base de datos
import mysql.connector
from mysql.connector import Error
from pandas import DataFrame

# Helpers
# -----------------------------------------------------------------------------
# Para obtener los tiempos de ejecución de procesos
def get_time(cond):
    if cond == "start":
        p = "Este proceso comenzó en "
    elif cond == "end":
        p = "Este proceso finalizó en "
    print("") 
    print(colored(p + str(datetime.now().strftime("%H:%M:%S")), "green", "on_white", attrs=["bold",'reverse', 'blink']))
    
    
def Filter(string, substr): 
        return [str for str in string if
                any(sub in str for sub in substr)] 
    
def NOTFilter(string, substr): 
    return [str for str in string if
            any(sub not in str for sub in substr)] 



###----------------------------- Capturando las URLs de los jugadores -----------------------

def content_string(href):
  url="https://www.transfermarkt.com" + href
  if url.find("/profil/spieler/") <= 0:
    url=''
  return url

def find_player_urls(team_url):
    
    get_time("start")
    
    tm_player_url = []
    tm_player_url_1 = []
    
    for completed,i in enumerate(team_url):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
        soup = BeautifulSoup(requests.get(i, headers=headers).content, "html.parser") 
        p = soup.find("table", class_ = "items").find_all("a")
        
        tm_player_url_1.append(pd.Series(list(map(lambda x: content_string(x["href"]), p))).unique().tolist())

        # Mensaje en pantalla: ¿cuántos equipos hemos scrapeado?
        sys.stdout.write("\r{0} equipos que acaban de ser scrapedo desde Transfermarkt!".format(completed+1))
        sys.stdout.flush()

    get_time("end")

    [tm_player_url.append(x) for x in tm_player_url_1 if x != '']

    tm_player_url  = pd.Series(list(itertools.chain(*tm_player_url))).unique().tolist()
    
    tm_player_url_df = pd.DataFrame({"TMURL":tm_player_url})
    tm_player_url_df["TMId"] = tm_player_url_df.TMURL.str.split("spieler/", expand = True)[1]

    get_time("start")
    new_tm_player_url = []

    for i in tm_player_url:
        if i != '':
          soup = BeautifulSoup(requests.get(i, headers=headers).content, "html.parser") 
          
          new_tm_player_url.append(i)

    get_time("end")

    tm_player_url_df = pd.DataFrame({"TMURL":new_tm_player_url})
    tm_player_url_df["TMId"] = tm_player_url_df.TMURL.str.split("spieler/", expand = True)[1]
    
    return tm_player_url_df.drop_duplicates()


team_url = ['https://www.transfermarkt.com/kaa-gent/startseite/verein/157']
tm_player_url_df = find_player_urls(team_url)
print("Cantidad de URLs de jugadores:",len(tm_player_url_df))

#tm_player_url_df


#-------------------------- Jugador Info -------------------------------

import pandas as pd

PlayerIdList = []
NameList = []
numberTshirtList = []
BirthList = []
AgeList = []
HeightList = []
CitizenshipList = []
PositionList = []
RolList = []
FootList = []
AgentList = []
teamIdList = []
CurrentClubList = []
JoinedList = []
ContractExpiresList = []
imageList = []

def return_Value(columnName,rows):
  value=''
  index = 0
  cantValues = len(rows)
  while (index < cantValues):
    if rows[index].text == columnName:
        value = rows[index + 1].text if rows[index + 1].text else ''
        value = value.lstrip()
        index = 200
    index = index + 1
  return value

def player_info(url):
    # PLAYER ID
    playerId = url.split("spieler/")[1]
    PlayerIdList.append(playerId)
    # Request
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    r=requests.get(url, headers = headers)
    soup=BeautifulSoup(r.content, "html.parser")
    
    #print(soup)
    try:
      #Nombre
      name = soup.find_all("title")
      #print(name)
      if(len(name)!=0):
        nameText = name[0].text
      else:
        nameText=url
        # image_url = soup.find_all("img", {"class": "data-header__profile-image"})
        # print(image_url)
        # nameText = image_url.attrs["title"]
        # print(nameText)

      titleNamePlayeer = nameText.split(' - ')
      namePlayeer = titleNamePlayeer[0] if titleNamePlayeer[0] else '-'
      NameList.append(namePlayeer)
      
      #Dorsal actual
      numberTshirt = soup.find_all("span", {"class": "data-header__shirt-number"})
      numberTshirt = numberTshirt[0].text if numberTshirt else '-'
      numberTshirt = numberTshirt.replace(" ", "").replace("\n", "")
      numberTshirtList.append(numberTshirt)

      #Club actual
      team = soup.find_all("span", {"itemprop": "affiliation"})
      try:
        teamId = team[0].find('a')['href'].split('verein/')[1] if len(team)>0 else '-'
      except:
        teamId ='-'
        pass
      teamIdList.append(teamId)
      CurrentClub = team[0].text if len(team)>0 else '-'
      CurrentClub = CurrentClub.replace("\n", "")
      CurrentClubList.append(CurrentClub)
      #aca
      #imagendata-header__profile-image
      #image_url = soup.find("img", class_ = "data-header__profile-image")
      image_url = soup.find_all("img", {"class": "data-header__profile-image"})
      if(len(image_url)!=0):
        imagePlayer = image_url[0]['src'] if image_url[0]['src'] else '-'
      else:
        imagePlayer = url
      imageList.append(imagePlayer)

      Values = soup.find_all("span", {"class": "info-table__content"})

      #print(Values[0].text) #column: Name in home country:
      #print(Values[1].text) #value: Marcos Javier Acuña
      #value = return_Value("Name in home country:",Values)
      #NameList.append(value)

      #print(Values[2].text) #column: Date of birth:
      #print(Values[3].text) #value: Oct 28, 1991 
      value = return_Value("Date of birth:",Values)
      BirthList.append(value if value else '-')
      #print(Values[4].text) #column: Place of birth:
      #print(Values[5].text) #value: Zapala
      #no nos interesa obtener la ciudad de nacimiento por ahora
      #print(Values[6].text) #column: Age:
      #print(Values[7].text) #value: 30
      value = return_Value("Age:",Values)
      AgeList.append(value if value else '-')
      #print(Values[8].text) #column: Height:
      #print(Values[9].text) #value: 1,72 m
      value = return_Value("Height:",Values)
      HeightList.append(value if value else '-')
      #print(Values[10].text) #column: Citizenship:
      #print(Values[11].text) #value: Argentina
      value = return_Value("Citizenship:",Values)
      CitizenshipList.append(value if value else '-')
      #print(Values[12].text) #column: Position:
      #print(Values[13].text) #value: Defender - Left-Back
      value = return_Value("Position:",Values)
      if value.find("Goalkeeper") > -1:
          value = "Goalkeeper - Goalkeeper"
      PositionRol = value.split(' - ')
      Rol = PositionRol[0] if len(PositionRol)>0 else '-'
      Position = PositionRol[1] if len(PositionRol)>1 else '-'
      PositionList.append(Position)
      RolList.append(Rol)
      #print(Values[14].text) #column: Foot:
      #print(Values[15].text) #value: left
      value = return_Value("Foot:",Values)
      FootList.append(value if value else '-')
      #print(Values[16].text) #column: Player agent:
      #print(Values[17].text) #value: Eleven Talent Group
      value = return_Value("Player agent:",Values)
      AgentList.append(value if value else '-')
      #print(Values[18].text) #column: Current club:
      #print(Values[19].text) #value: Sevilla FC
      #value = return_Value("Current club:",Values)
      #CurrentClubList.append(value)
      #print(Values[20].text) #column: Joined:
      #print(Values[21].text) #value: Sep 14, 2020
      value = return_Value("Joined:",Values)
      JoinedList.append(value if value else '-')
      #print(Values[22].text) #column: Contract expires:
      #print(Values[22].text) #value: Jun 30, 2024
      value = return_Value("Contract expires:",Values)
      ContractExpiresList.append(value if value else '-')
      
      return
    except Error as e:
      print('Hubo un error')
      objeto_log = {'page' : 'Tranfermarket', 
                'metodo' : 'player_info', 
                'fecha': datetime.now(), 
                'error' : str(e), 
                'url' : url
                }
      print(objeto_log)
      pass
    
# Test
#player_info("https://www.transfermarkt.com/daniele-de-rossi/profil/spieler/5947") 
# si deseamos obtener todos los datos de cada jugador que obtuvimos el link debemos descomentar estas líneas

#Test
# urls =['https://www.transfermarkt.com/exequiel-palacios/profil/spieler/401578',
#        'https://www.transfermarkt.com/santiago-garcia/profil/spieler/281405',
#        'https://www.transfermarkt.com/santiago-garcia/profil/spieler/576028',
#        'https://www.transfermarkt.com/santiago-garcia/profil/spieler/315063']

urls_all = tm_player_url_df['TMURL']
urls=[]
for element in urls_all:
    if element not in urls:
        urls.append(element)  
get_time("start")    
for i in urls:
  
  player_info(i) 
get_time("end")
#print(PlayerIdList)
df_player = pd.DataFrame({"playerId":PlayerIdList
                   ,"name":NameList
                   ,"tShirt":numberTshirtList
                   ,"birth":BirthList
                   ,"age":AgeList
                   ,"height":HeightList
                   ,"citizenship":CitizenshipList
                   ,"position":PositionList
                   ,"rol":RolList
                   ,"foot":FootList
                   ,"agen":AgentList
                   ,"teamId":teamIdList
                   ,"currentClub":CurrentClubList
                   ,"joined":JoinedList
                   ,"contractExpires":ContractExpiresList
                   ,"image":imageList})

# # Formato de fecha en Inglés. Pero podemos cambiarlo a formato tipo date
df_player['birth']=pd.to_datetime(df_player['birth'], dayfirst=True, errors = "coerce")
df_player = df_player[df_player.birth.notnull()]

# using now() to get current time
current_time = datetime.now()
#agreguemos una columna más que nos informe cuándo fue la fecha de extracción de estos datos
df_player['dateExtraction'] = current_time;

#df_player


df_posiciones= df_player[['name',	'currentClub', 'contractExpires','position']]

df_portero =df_posiciones[df_posiciones.position == 'Goalkeeper']

now = datetime.now()
print(now.date())

date_obj = datetime.strptime('2022-03-10', '%Y-%m-%d').date()
print(date_obj)

color = '';
if ((date_obj-now.date()).days > 60):
  color = 'b';
elif ((date_obj-now.date()).days > 30):
  color = 'y';
else:
  color = 'r';

#formato de fecha yyyy-mm-dd
from datetime import date

def color_contrato(fecha):
  #print(fecha)
  now = date.today()
  color = '';
  try:
    date_obj = datetime.strptime(str(fecha[:10]), '%Y-%m-%d').date()
    date_obj = date(date_obj.year, date_obj.month, date_obj.day)
    #print((date_obj - now).days)
    if ((date_obj - now).days > 900):
      color = '#2169CC';
    elif ((date_obj - now).days > 400):
      color = '#FFC300';
    else:
      color = '#D63B15';
  except:
      color = 'w';
  #print(color)
  return color

def convertToDatetime(txtDate):
  rtaDate=''
  try:
    rtaDate=pd.to_datetime(txtDate, dayfirst=True, errors = "coerce")
    if(pd.isnull(rtaDate)):
      rtaDate='-'
  except:
    rtaDate='-'
  return rtaDate

df_posiciones['dateContractExpires']=df_posiciones['contractExpires'].apply(lambda x: convertToDatetime(x))


pitch = Pitch(pitch_color='grass', stripe=True)
fig, ax = pitch.draw(figsize=(16, 8))

#ESTA ES LA QUE ME PASASTE TU Y NO ME VA BIEN 
X_Jugador=-3
Y_Jugador=35
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Goalkeeper'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=25
Y_Jugador=35
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Centre-Back'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)
    
X_Jugador=25
Y_Jugador=2
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Left-Back'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=25
Y_Jugador=67
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Right-Back'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=85
Y_Jugador=67
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Right Winger'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=85
Y_Jugador=2
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Left Winger'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=48
Y_Jugador=20
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Central Midfield'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=52
Y_Jugador=50
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Defensive Midfield'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=67
Y_Jugador=35
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Attacking Midfield'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=90
Y_Jugador=22
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Second Striker'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)

X_Jugador=95
Y_Jugador=45
for ind in df_posiciones.index:
  if (df_posiciones['position'][ind].rstrip() == 'Centre-Forward'):
    Y_Jugador=Y_Jugador+3
    nombre = df_posiciones['name'][ind]
    color = color_contrato(str(df_posiciones['dateContractExpires'][ind]))
    vencimiento = df_posiciones['contractExpires'][ind]
    textJugador = nombre + ' (' + str(vencimiento) + ')'
    plt.annotate(textJugador, xy=(X_Jugador,Y_Jugador), backgroundcolor=color)
