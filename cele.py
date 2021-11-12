# %% PAQUETES
import numpy as np
import pandas as pd
from pathlib import Path
import re
import os
import networkx as nx
from collections import Counter
# tweeter processing
# Import all needed libraries
import tweepy                   # Python wrapper around Twitter API
import json
import csv
from datetime import date
from datetime import datetime
import time
from tqdm import tqdm
import webbrowser
import string

# %% PATHS
base_dir = Path.cwd()
data_dir = base_dir / 'data'
#Create if they don't exist
Path(data_dir).mkdir(parents=True, exist_ok=True)

# %% FUNCTIONS
#sheet_names=['Búsqueda español', 'Búsqueda portugués']
# dfs_= []
# for sh in sheet_names:
#     df_ = pd.read_excel(data_dir/'2.xlsx', sheet_name=sh)
#     df_['sheet'] = sh.split(' ')[-1]
#     dfs_.append(df_)
#
# df = pd.concat(dfs_)
df = pd.read_excel(data_dir/'2.xlsx')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.rename(columns={'Influencer':'screen_name', 'Twitter Followers':'followers'})

# %% RT and MENTIONS
def get_RT(text):
    try:
        value= re.findall('(?<=RT\s)(@[\w]+[\w]+)', text)[0]
    except:
        value = np.nan
    return value

def get_QT(text):
    try:
        value= re.findall('(?<=QT\s)(@[\w]+[\w]+)', text)[0]
    except:
        value = np.nan
    return value

def get_mentions(text):
    try:
        value= re.findall('(?<!RT\s)(?<!QT\s)(@[\w]+[\w]+)', text)
    except:
        value = np.nan

    return value

def get_HT(text):
    return re.findall('(#[\w+]+[\w+]+)', text)

df['RT'] = df['Hit Sentence'].apply(lambda x: get_RT(x))
df['QT'] = df['Hit Sentence'].apply(lambda x: get_QT(x))
df['M'] = df['Hit Sentence'].apply(lambda x: get_mentions(x))
df['HT']= df['Hit Sentence'].apply(lambda x: get_HT(x))

# %% LOWER CASE USEFULL COLUMNS
df['screen_name'] = df['screen_name'].str.lower()  #lower case
df.M = df.M.apply(lambda x: [s.lower() for s in x])

# %% Cambio de las x por cosas entendibles
x_columns = ['EVENTO','Competencia', 'Modelo', 'Diseño', 'Transparencia', 'Procesos',
             'Corregulación', 'Consumidor', 'Responsabilidad', 'Utilidad',
             'Comportamiento', 'Informes, rendición',
             'Políticas, toma de decisiones', 'Riesgos, daños', 'Algoritmos']
df[x_columns] =df[x_columns].replace(' ', np.nan).replace(['x','X','A'],1).fillna(0).astype(int)

# %% productores sin RT
with pd.ExcelWriter(data_dir/'productores.xlsx') as writer:
    for key, df_ in zip(['no_RT','todo'], [df.loc[(df.RT.isnull())], df]):
        prod = df_.groupby(by='screen_name')\
                 .agg({'URL':'count', 'Engagement':'sum', 'followers':'max', 'EVENTO':'sum','Competencia':'sum', 'Modelo':'sum', 'Diseño':'sum',
                       'Transparencia':'sum', 'Procesos':'sum', 'Corregulación':'sum', 'Consumidor':'sum',
                       'Responsabilidad':'sum', 'Utilidad':'sum','Comportamiento':'sum', 'Informes, rendición':'sum',
                       'Políticas, toma de decisiones':'sum', 'Riesgos, daños':'sum', 'Algoritmos':'sum'})\
                 .sort_values(by=['URL','Engagement'], ascending=False).rename(columns={'URL':'count','Engagement':'engagement_sum'})\
                 .reset_index()
        prod['engagement_mean'] = prod['engagement_sum'] / prod['count']

        prod = prod[prod.columns.insert(3,'engagement_mean')[:-1]]
        prod.to_excel(writer, sheet_name='Base', index=False, encoding='utf8', sheet_name = key)

# %% HTS
with pd.ExcelWriter(data_dir/'hashtags.xlsx') as writer:
    for sh in df.sheet.unique():
        df_ = df.loc[df.sheet == sh]
        counter = Counter(df_.HT.sum())

        HTs = pd.DataFrame.from_records(list(dict(counter).items()), columns=['HT','count'])\
                .sort_values(by='count',ascending=False)\
                .head(100)

        HTs.to_excel(writer, sheet_name=sh, index=False, encoding='utf8')

# %% EDGES
dfs_ =[]
for col in ['M']:
    if col == 'M':
        df_ = df[['screen_name','M']].explode('M').dropna()
        df_.M = df_.M.replace({'@':''}, regex=True)
    else:
        df_ = df[['screen_name',col]].dropna()

    df_.screen_name = df_.screen_name.replace({'@':''}, regex=True) #solo para este caso
    df_.rename(columns={'screen_name':'Source', col:'Target'}, inplace=True)
    df_['Interaction'] = col

    dfs_.append(df_)

edges = pd.concat(dfs_, ignore_index=True)
edges['Weight'] = 1
edges = edges.groupby(by=['Source','Target','Interaction']).count().reset_index()
edges.shape

# %% NODES
def get_nodes(edges):
    handles = pd.unique(edges[['Source','Target']].values.ravel('K'))
    ids = range(len(handles)) # Irrelevant ID

    nodes = pd.DataFrame({'Id':ids,'Handles':handles})

    nodes = pd.merge(nodes, df[['screen_name','followers']].replace({'@':''}, regex=True), how='left', left_on='Handles', right_on='screen_name')
    nodes.drop(columns=['screen_name'], inplace=True)
    nodes = nodes.sort_values(by=['Id','followers']).drop_duplicates(subset='Id', keep='last')

    return nodes

nodes = get_nodes(edges)
nodes.shape

# %% GET FOLLOWERS
def get_api():
    consumer_key = 'eIVPmXDdEYEtvfJaPLtJD12iV'
    consumer_secret = 'aOnHd3rwXOgL2hsmuOXVtGP51NBePNHmovGDRGnKGQSwN6iMQZ'
    access_token = '1386719422971355137-pd3S7pBWRLEjXwzI1qRz67Sz9txP9I'
    access_token_secret = 'qYqySGmFSJGNVlZYSvr02kjyMmxMWOuORKCKglL5T3C5p'
    callback_uri = 'oob' # https://cfe.sh/twitter/callback
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret,callback_uri)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    return api

api = get_api()

def get_followers(screen_name):
    try:
        user = api.get_user(screen_name=screen_name)
        return user.followers_count
    except:
        return 0

if (data_dir/'nodes_complete.pkl').is_file():
    nodes = pd.read_pickle(data_dir/'nodes_complete.pkl')
else:
    nodes = get_nodes(edges)
    for i in tqdm(nodes.loc[nodes.followers.isnull()].index):
        screen_name= nodes['Handles'][i]
        nodes['followers'][i] = get_followers(screen_name)
    nodes.to_pickle(data_dir/'nodes_complete.pkl')

# %% CHANGE EDGES TO IDS
for c in ['Source', 'Target']:
        edges[c] = edges[c].map(dict(nodes[['Handles','Id']].values))

edges[['Source','Target']] = edges[['Source','Target']].astype(int)

# %% NOT FULLY CONNECTED
edges.to_csv(data_dir/'networks'/'nfc'/'edges.csv', index=False)
nodes.to_csv(data_dir/'networks'/'nfc'/'nodes.csv', index=False)
print(edges.shape)

# %% FULLY CONNECTED FILTER
def get_fully_connected(nodes,edges):
    D = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr=True, create_using=nx.DiGraph)
    nodes.index = nodes.Id # Toca hacer esto para que se puedan recorrer los nodos al agragarlos a la red
    for i in sorted(D.nodes()):
        D.nodes[i]['Handles'] = nodes.Handles[i]

    largest_cc = max(nx.weakly_connected_components(D), key=len)
    D_connected = D.subgraph(largest_cc)

    nodes = pd.DataFrame.from_dict(dict(D_connected.nodes(True)), orient='index').reset_index().rename(columns={'index':'Id'})
    edges = nx.to_pandas_edgelist(D_connected).rename(columns={'source':'Source', 'target':'Target'})

    return nodes, edges

for tipo in ['M']:
    nodes_, edges_ = get_fully_connected(nodes,edges.loc[edges.Interaction == tipo])
    edges_.to_csv(data_dir/'networks'/'fc'/('edges_'+ tipo +'.csv'), index=False)
    nodes.to_csv(data_dir/'networks'/'fc'/'nodes.csv', index=False)
    print(edges_.shape)
