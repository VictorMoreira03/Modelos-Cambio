import streamlit as st
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 

st.set_page_config(page_title="EME Currencies Analysis", layout="wide")

def main(): 
   
   
    tickers_EME = {
    'USDCNY' : 'CNY=X',
    'USDMXN' : 'MXN=X',
    'USDKRW' : 'KRW=X',
    'USDINR' : 'INR=X',
    'USDBRL' : 'BRL=X',
    'USDTWD' : 'TWD=X',
    'USDSGD' : 'SGD=X',
    'USDHKD' : 'HKD=X',
    'USDVND' : 'VND=X',
    'USDMYR' : 'MYR=X',
    'USDTHB' : 'THB=X',
    'USDILS' : 'ILS=X',
    'USDIDR' : 'IDR=X',
    'USDPHP' : 'PHP=X',
    'USDCLP' : 'CLP=X',
    'USDCOP' : 'COP=X',
    'USDSAR' : 'SAR=X', 
    'USDARS' : 'ARS=X',
    'USDRUB' : 'RUB=X'
    }
    
    weights_EME = {
    'USDCNY' : 31.3,
    'USDMXN' : 25.7,
    'USDKRW' : 6.6,
    'USDINR' : 5.3,
    'USDBRL' : 3.9,
    'USDTWD' : 3.8,
    'USDSGD' : 3.1,
    'USDHKD' : 2.9,
    'USDVND' : 2.6,
    'USDMYR' : 2.4,
    'USDTHB' : 2.2,
    'USDILS' : 2.1,
    'USDIDR' : 1.3,
    'USDPHP' : 1.3,
    'USDCLP' : 1.2,
    'USDCOP' : 1.1,
    'USDSAR' : 1.1, 
    'USDARS' : 1.1,
    'USDRUB' : 1,
    'EME_Index' : 100
    }

    windows = {
        'D1': 1,
        'Week': 5,
        '2 Weeks': 10,
        'Month': 21,
        '3 Months': 63,
        '6 Months': 126,
        '12 Months': 252
    }

    data = yf.download(tickers=list(tickers_EME.values()),
                       period='19y',
                       interval='1d')['Close']

    df_EME = pd.DataFrame()

    for asset, ticker in tickers_EME.items(): 
        df_EME[asset] = data[ticker]

    # Exemplo: df_EME possui várias colunas, cada uma representando uma moeda, ex: ['USDCNY','USDMXN','...']
    # threshold = 0.3 (30%)
    threshold = 0.3

    # 1) Para cada coluna de df_EME, calculamos variação diária e detectamos outliers
    for col in df_EME.columns:
        # Calcula variação percentual diária
        ret = df_EME[col].pct_change()
        
        # Cria uma máscara booleana para dias com variação acima de +30% ou abaixo de -30%
        outliers = ret.abs() > threshold
        
        # Substitui no df_EME o valor do dia 't' por NaN quando ret(t) é outlier
        # (df.loc[linhas, col] = np.nan)
        df_EME.loc[outliers, col] = np.nan

    # 2) Fazer forward fill ou outra correção
    # Substitui as ocorrências de NaN pelo valor anterior
    df_EME = df_EME.ffill()


    ## Construindo o índice EME via lógica Logarítmica (média geométrica)
    #1) Calculo log(prices) de cada par
    #2) Calculo o log-return a cada hora: Ln(Pt) - Ln(Pt-1)
    #3) Somo os log-returns com os devidos pesos normalizados
    #4) Soma cumulativa para obter ln(índice)
    #5) Exponenciar para ter o valor do índice 



    #1)  pesos normalizados (para somarem 1.0)
    total_weight = sum(weights_EME.values())
    weights_norm = {k: v/total_weight for k, v in weights_EME.items()}

    #2) log dos preços (df do mesmo shape de df_EME)
    log_prices = np.log(df_EME)

    #3) Log-returns horário (diferença horária)
    log_returns = log_prices.diff()

    #4) Soma ponderada dos log-returns a cada hora
    # Criar uma Series "weighted_log_ret" que, para cada timestamp, some w_i*log_returns[i]
    weighted_log_ret = pd.Series(0.0, index = log_returns.index)
    for col in log_returns.columns:
        w = weights_norm[col]
        weighted_log_ret += w * log_returns[col]


    #5) Índice do dólar vs. EME (cumulativo de log return, base 100)
    #   EME_index(0) = 100 => EME_index(t) = 100*exp(soma cumulativa de log_return até t)
    log_index = weighted_log_ret.cumsum()
    eme_index = 100 * np.exp(log_index)

    df_EME['EME_Index'] = eme_index
    # Agora "eme_index" é uma Series com o valor do índice do dólar vs. essas moedas EME ao longo do tempo (horário)

    df_EME1 = df_EME[['EME_Index']].ffill()

    df_EME1['EME_Index_200d'] = df_EME1['EME_Index'].rolling(window=200).mean()
    df_EME1['EME_Index_50d'] = df_EME1['EME_Index'].rolling(window=50).mean()

    df_plot_EME = df_EME1[['EME_Index','EME_Index_200d','EME_Index_50d']]

    fig_EME = go.Figure()

    fig_EME.add_trace(go.Scatter(
        x=df_plot_EME.index,
        y=df_plot_EME["EME_Index"],
        mode="lines",
        name="EME",                      # Legenda na chart
        line=dict(color="black")          # Define a cor da linha
    ))

    fig_EME.add_trace(go.Scatter(
        x=df_plot_EME.index,
        y=df_plot_EME["EME_Index_200d"],
        mode="lines",
        name="200d Média",
        line=dict(color="red")           # Define a cor para a média móvel
    ))

    fig_EME.add_trace(go.Scatter(
        x=df_plot_EME.index,
        y=df_plot_EME["EME_Index_50d"],
        mode="lines",
        name="50d Média",
        line=dict(color="blue")           # Define a cor para a média móvel
    ))


    fig_EME.update_layout(
        title="Performance do EME (19 anos)",
        yaxis=dict(range=[85, 130]),
        xaxis=dict(range=[
            df_plot_EME.index[0], 
            df_plot_EME.index[-1] + pd.Timedelta("40d")
        ])
    )

    st.write("### EME Index")
    st.plotly_chart(fig_EME, use_container_width = True)


    # Montando heatmap
    assets_EME = ['USDCNY','USDMXN','USDKRW','USDINR','USDBRL','USDTWD','USDSGD','USDHKD','USDVND','USDMYR','USDTHB','USDILS','USDIDR','USDPHP','USDCLP','USDCOP','USDSAR','USDARS','USDRUB','EME_Index']
    heatmap_EME_data = pd.DataFrame(index=assets_EME, columns=windows.keys())

    for asset_EME in assets_EME: 
        for label, w in windows.items():
            #Pega o valor mais recente e compara com w dias
            try: 
                recent_price = df_EME[asset_EME].iloc[-1]
                old_price = df_EME[asset_EME].iloc[-(1+w)]
                log_return = (np.log(recent_price) - np.log(old_price))*100
                # ret_simples = (np.exp(log_return)-1)*100
                # change_pct = ((recent_price / old_price) - 1)*100
                heatmap_EME_data.loc[asset_EME,label] = log_return
            except:
                heatmap_EME_data.loc[asset_EME,label] = np.nan



    # Adicionar pesos
    heatmap_EME_data['Weight (%)'] = heatmap_EME_data.index.map(weights_EME)
    cols = ['Weight (%)'] + [c for c in heatmap_EME_data.columns if c != 'Weight (%)']
    heatmap_EME_data = heatmap_EME_data[cols]

    # Formatação
    cols_for_color = [c for c in heatmap_EME_data.columns if c != 'Weight (%)']
    for c in cols_for_color:
        heatmap_EME_data[c] = pd.to_numeric(heatmap_EME_data[c], errors='coerce')
    abs_max = np.abs(heatmap_EME_data[cols_for_color].values).max()

    # Exibir como DataFrame normal
    st.write("### Heatmap Emerging Currencies")
    st.dataframe(heatmap_EME_data.style
                 .background_gradient(cmap='RdBu',
                                      subset=cols_for_color,
                                      vmin=-abs_max,
                                      vmax=abs_max)
                 .format("{:+.2f}%", subset=cols_for_color)
                 .format("{:.1f}%", subset=['Weight (%)'])
                )


    tickers_EME_ExC = {
    'USDMXN' : 'MXN=X',
    'USDKRW' : 'KRW=X',
    'USDINR' : 'INR=X',
    'USDBRL' : 'BRL=X',
    'USDTWD' : 'TWD=X',
    'USDSGD' : 'SGD=X',
    'USDHKD' : 'HKD=X',
    'USDVND' : 'VND=X',
    'USDMYR' : 'MYR=X',
    'USDTHB' : 'THB=X',
    'USDILS' : 'ILS=X',
    'USDIDR' : 'IDR=X',
    'USDPHP' : 'PHP=X',
    'USDCLP' : 'CLP=X',
    'USDCOP' : 'COP=X',
    'USDSAR' : 'SAR=X', 
    'USDARS' : 'ARS=X',
    'USDRUB' : 'RUB=X'
    }
    
    weights_EME_ExC = {
    'USDMXN' : 37.4,
    'USDKRW' : 9.6,
    'USDINR' : 7.7,
    'USDBRL' : 5.7,
    'USDTWD' : 5.5,
    'USDSGD' : 4.5,
    'USDHKD' : 4.2,
    'USDVND' : 3.8,
    'USDMYR' : 3.5,
    'USDTHB' : 3.2,
    'USDILS' : 3.1,
    'USDIDR' : 1.9,
    'USDPHP' : 1.9,
    'USDCLP' : 1.7,
    'USDCOP' : 1.6,
    'USDSAR' : 1.6, 
    'USDARS' : 1.6,
    'USDRUB' : 1.5,
    'EME_Index_ExC' : 100
    }

    windows = {
        'D1': 1,
        'Week': 5,
        '2 Weeks': 10,
        'Month': 21,
        '3 Months': 63,
        '6 Months': 126,
        '12 Months': 252
    }

    data_ExC = yf.download(tickers=list(tickers_EME_ExC.values()),
                       period='19y',
                       interval='1d')['Close']

    df_EME_ExC = pd.DataFrame()

    for asset, ticker in tickers_EME_ExC.items(): 
        df_EME_ExC[asset] = data_ExC[ticker]


    # Exemplo: df_EME_ExC possui várias colunas, cada uma representando uma moeda, ex: ['USDCNY','USDMXN','...']
    # threshold = 0.3 (30%)
    threshold = 0.2

    # 1) Para cada coluna de df_EME_ExC, calculamos variação diária e detectamos outliers
    for col in df_EME_ExC.columns:
        # Calcula variação percentual diária
        ret = df_EME_ExC[col].pct_change()
        
        # Cria uma máscara booleana para dias com variação acima de +30% ou abaixo de -30%
        outliers = ret.abs() > threshold
        
        # Substitui no df_EME_ExC o valor do dia 't' por NaN quando ret(t) é outlier
        # (df.loc[linhas, col] = np.nan)
        df_EME_ExC.loc[outliers, col] = np.nan

    # 2) Fazer forward fill ou outra correção
    # Substitui as ocorrências de NaN pelo valor anterior
    df_EME_ExC = df_EME_ExC.ffill()


    ## Construindo o índice EME via lógica Logarítmica (média geométrica)
    #1) Calculo log(prices) de cada par
    #2) Calculo o log-return a cada hora: Ln(Pt) - Ln(Pt-1)
    #3) Somo os log-returns com os devidos pesos normalizados
    #4) Soma cumulativa para obter ln(índice)
    #5) Exponenciar para ter o valor do índice 



    #1)  pesos normalizados (para somarem 1.0)
    total_weight = sum(weights_EME_ExC.values())
    weights_norm = {k: v/total_weight for k, v in weights_EME_ExC.items()}

    #2) log dos preços (df do mesmo shape de df_EME)
    log_prices = np.log(df_EME_ExC)

    #3) Log-returns horário (diferença horária)
    log_returns = log_prices.diff()

    #4) Soma ponderada dos log-returns a cada hora
    # Criar uma Series "weighted_log_ret" que, para cada timestamp, some w_i*log_returns[i]
    weighted_log_ret = pd.Series(0.0, index = log_returns.index)
    for col in log_returns.columns:
        w = weights_norm[col]
        weighted_log_ret += w * log_returns[col]


    #5) Índice do dólar vs. EME (cumulativo de log return, base 100)
    #   EME_index(0) = 100 => EME_index(t) = 100*exp(soma cumulativa de log_return até t)
    log_index = weighted_log_ret.cumsum()
    eme_ExC_index = 100 * np.exp(log_index)

    df_EME_ExC['EME_Index_ExC'] = eme_ExC_index
    # Agora "eme_index" é uma Series com o valor do índice do dólar vs. essas moedas EME ao longo do tempo (horário)

    df_EME1_ExC = df_EME_ExC[['EME_Index_ExC']].ffill()

    df_EME1_ExC['EME_Index_ExC_200d'] = df_EME1_ExC['EME_Index_ExC'].rolling(window=200).mean()
    df_EME1_ExC['EME_Index_ExC_50d'] = df_EME1_ExC['EME_Index_ExC'].rolling(window=50).mean()

    df_plot_EME_ExC = df_EME1_ExC[['EME_Index_ExC','EME_Index_ExC_200d','EME_Index_ExC_50d']]


    fig_EME_ExC = go.Figure()

    fig_EME_ExC.add_trace(go.Scatter(
        x=df_plot_EME_ExC.index,
        y=df_plot_EME_ExC["EME_Index_ExC"],
        mode="lines",
        name="EME_ExC",                      # Legenda na chart
        line=dict(color="blue")          # Define a cor da linha
    ))

    fig_EME_ExC.add_trace(go.Scatter(
        x=df_plot_EME_ExC.index,
        y=df_plot_EME_ExC["EME_Index_ExC_200d"],
        mode="lines",
        name="200d Média",
        line=dict(color="red")           # Define a cor para a média móvel
    ))

    fig_EME_ExC.update_layout(
        title="Performance do EME_ExC (10 anos)",
        yaxis=dict(range=[85, 140]),
        xaxis=dict(range=[
                df_plot_EME_ExC.index[0], 
                df_plot_EME_ExC.index[-1] + pd.Timedelta("40d")
        ])
    )

    st.write("### EME_ExC Index")
    st.plotly_chart(fig_EME_ExC, use_container_width = True)

if __name__ == "__main__":
    main()
