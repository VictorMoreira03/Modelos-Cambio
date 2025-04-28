import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import coint


def rodar_modelo(modelo="BEER"):
    if modelo == "BEER":
        rodar_BEER()

def rodar_BEER():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=csv"
    df_cambio = pd.read_csv(url, sep=";", parse_dates=["data"])
    df_cambio.rename(columns={"valor": "USD_BRL"}, inplace=True)
    df_cambio_m = data.resample('M').mean()



def dolar_mean():
    # URL da API do Banco Central (série 1: USD/BRL)
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=csv"

    # Baixar e carregar os dados
    df_cambio = pd.read_csv(url, sep=";", parse_dates=["data"])

    # Renomear coluna para facilitar
    df_cambio.rename(columns={"valor": "USD_BRL"}, inplace=True)

    # Converter a coluna "USD_BRL" para numérico (caso esteja como string)
    df_cambio["USD_BRL"] = df_cambio["USD_BRL"].astype(str).str.replace(",", ".")
    df_cambio["USD_BRL"] = pd.to_numeric(df_cambio["USD_BRL"], errors='coerce')

    # Definir a coluna "data" como índice
    df_cambio.set_index("data", inplace=True)

    # Filtrar somente os dados após 1999
    df_cambio = df_cambio.loc[df_cambio.index >= "1999-01-01"]

    # Calcular a média mensal da taxa de câmbio
    df_cambio_m = df_cambio.resample("M").mean()

    
def dolar_last():
    # URL da API do Banco Central (série 1: USD/BRL)
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=csv"

    # Baixar e carregar os dados
    df_cambio = pd.read_csv(url, sep=";", parse_dates=["data"])

    # Renomear coluna para facilitar
    df_cambio.rename(columns={"valor": "USD_BRL"}, inplace=True)

    # Converter a coluna "USD_BRL" para numérico (caso esteja como string)
    df_cambio["USD_BRL"] = df_cambio["USD_BRL"].astype(str).str.replace(",", ".")
    df_cambio["USD_BRL"] = pd.to_numeric(df_cambio["USD_BRL"], errors='coerce')

    # Definir a coluna "data" como índice
    df_cambio.set_index("data", inplace=True)

    # Filtrar somente os dados após 1999
    df_cambio = df_cambio.loc[df_cambio.index >= "1999-01-01"]

    # Calcular a média mensal da taxa de câmbio
    df_cambio_m = df_cambio.resample("M").last()

    


## ----------------------------- MAIN -------------------------------##

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=csv"
df_cambio = pd.read_csv(url, sep=";", parse_dates=["data"])
df_cambio.rename(columns={"valor": "USD_BRL"}, inplace=True)
df_cambio_m = data.resample('M').mean()
