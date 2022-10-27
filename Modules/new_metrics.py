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
