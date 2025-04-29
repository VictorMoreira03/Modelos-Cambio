
 
import streamlit as st
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 



st.set_page_config(page_title="Broad Currencies Analysis")


def main(): 

    tickers_BROAD = {
    'USDEUR' : 'EUR=X',
    'USDCNY' : 'CNY=X',
    'USDCAD' : 'CAD=X',
    'USDMXN' : 'MXN=X',
    'USDJPY' : 'JPY=X',
    'USDGBP' : 'GBP=X',
    'USDKRW' : 'KRW=X',
    'USDINR' : 'INR=X',
    'USDCHF' : 'CHF=X',
    'USDBRL' : 'BRL=X',
    'USDTWD' : 'TWD=X',
    'USDSGD' : 'SGD=X',
    'USDHKD' : 'HKD=X',
    'USDAUD' : 'AUD=X',
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
    'USDRUB' : 'RUB=X',
    'USDSEK' : 'SEK=X',
    }
    
    weights_BROAD = {
    'USDEUR': 18.6,
    'USDCNY' : 16.2,
    'USDCAD': 13.6,
    'USDMXN' : 13.3,
    'USDJPY': 6.4,
    'USDGBP': 5.1,
    'USDKRW' : 3.4,
    'USDINR' : 2.7,
    'USDCHF': 2.7,
    'USDBRL' : 2,
    'USDTWD' : 2,
    'USDSGD' : 1.6,
    'USDHKD' : 1.5,
    'USDAUD' : 1.4,
    'USDVND' : 1.3,
    'USDMYR' : 1.3,
    'USDTHB' : 1.1,
    'USDILS' : 1.1,
    'USDIDR' : 0.7,
    'USDPHP' : 0.7,
    'USDCLP' : 0.6,
    'USDCOP' : 0.6,
    'USDSAR' : 0.6, 
    'USDARS' : 0.5,
    'USDRUB' : 0.5,
    'USDSEK': 0.5,
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

    data_BROAD = yf.download(tickers=list(tickers_BROAD.values()),
                       period='19y',
                       interval='1d')['Close']

    df_BROAD = pd.DataFrame()

    for asset_BROAD, ticker in tickers_BROAD.items(): 
        df_BROAD[asset_BROAD] = data_BROAD[ticker]


    # Exemplo: df_BROAD possui várias colunas, cada uma representando uma moeda, ex: ['USDCNY','USDMXN','...']
    # threshold = 0.3 (30%)
    threshold = 0.2

    # 1) Para cada coluna de df_BROAD, calculamos variação diária e detectamos outliers
    for col in df_BROAD.columns:
        # Calcula variação percentual diária
        ret = df_BROAD[col].pct_change()
        
        # Cria uma máscara booleana para dias com variação acima de +30% ou abaixo de -30%
        outliers = ret.abs() > threshold
        
        # Substitui no df_BROAD o valor do dia 't' por NaN quando ret(t) é outlier
        # (df.loc[linhas, col] = np.nan)
        df_BROAD.loc[outliers, col] = np.nan

    # 2) Fazer forward fill ou outra correção
    # Substitui as ocorrências de NaN pelo valor anterior
    df_BROAD = df_BROAD.ffill()

    ## Construindo o índice EME via lógica Logarítmica (média geométrica)
    #1) Calculo log(prices) de cada par
    #2) Calculo o log-return a cada hora: Ln(Pt) - Ln(Pt-1)
    #3) Somo os log-returns com os devidos pesos normalizados
    #4) Soma cumulativa para obter ln(índice)
    #5) Exponenciar para ter o valor do índice 



    #1)  pesos normalizados (para somarem 1.0)
    total_weight = sum(weights_BROAD.values())
    weights_norm = {k: v/total_weight for k, v in weights_BROAD.items()}

    #2) log dos preços (df do mesmo shape de df_EME)
    log_prices = np.log(df_BROAD)

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
    BROAD_index = 100 * np.exp(log_index)

    df_BROAD['BROAD_Index'] = BROAD_index
    # Agora "eme_index" é uma Series com o valor do índice do dólar vs. essas moedas EME ao longo do tempo (horário)

    df_BROAD1 = df_BROAD[['BROAD_Index']].ffill()

    df_BROAD1['BROAD_Index_200d'] = df_BROAD1['BROAD_Index'].rolling(window=200).mean()
    df_BROAD1['BROAD_Index_50d'] = df_BROAD1['BROAD_Index'].rolling(window=50).mean()

    df_plot_BROAD = df_BROAD1[['BROAD_Index','BROAD_Index_200d','BROAD_Index_50d']]

    # fig, ax = plt.subplots(figsize=(20, 10))

    # ax.plot(df_plot.index, df_plot['BROAD_Index'], label='BROAD Index')
    # ax.plot(df_plot.index, df_plot['BROAD_Index_200d'], label='200d Média', linestyle='--')

    # # Define os limites do eixo Y de 80 a 120
    # ax.set_ylim([80, 145])
    # ax.set_xlim([df_plot.index[0],df_plot.index[-1]+pd.Timedelta('40d')])

    # ax.set_xlabel('Data')
    # ax.set_ylabel('Índice')
    # ax.set_title('Performance do BROAD Index (10 anos)')
    # ax.legend()

    # # Exibe no Streamlit
    # st.write("### BROAD Index")
    # st.pyplot(fig)

    fig_BROAD = go.Figure()

    fig_BROAD.add_trace(go.Scatter(
        x=df_plot_BROAD.index,
        y=df_plot_BROAD["BROAD_Index"],
        mode="lines",
        name="BROAD Dollar",                      # Legenda na chart
        line=dict(color="blue")          # Define a cor da linha
    ))

    fig_BROAD.add_trace(go.Scatter(
        x=df_plot_BROAD.index,
        y=df_plot_BROAD["BROAD_Index_200d"],
        mode="lines",
        name="200d Média",
        line=dict(color="red")           # Define a cor para a média móvel
    ))

    fig_BROAD.update_layout(
        title="Performance do BROAD Dollar",
        yaxis=dict(range=[85, 150]),
        xaxis=dict(range=[
            df_plot_BROAD.index[0], 
            df_plot_BROAD.index[-1] + pd.Timedelta("40d")
        ])
    )

    st.write("### BROAD Dollar Index")
    st.plotly_chart(fig_BROAD, use_container_width = True)


    # Montando heatmap
    assets_BROAD = ['USDEUR','USDCNY','USDCAD','USDMXN','USDJPY','USDGBP','USDKRW','USDINR','USDCHF','USDBRL','USDTWD','USDSGD','USDHKD','USDAUD','USDVND','USDMYR','USDTHB','USDILS','USDIDR','USDPHP','USDCLP','USDCOP','USDSAR','USDARS','USDRUB','USDSEK','BROAD_Index']
    heatmap_BROAD_data = pd.DataFrame(index=assets_BROAD, columns=windows.keys())

    for asset_BROAD in assets_BROAD: 
        for label, w in windows.items():
            #Pega o valor mais recente e compara com w dias
            try: 
                recent_price = df_BROAD[asset_BROAD].iloc[-1]
                old_price = df_BROAD[asset_BROAD].iloc[-(1+w)]
                log_return = (np.log(recent_price) - np.log(old_price))*100
                # ret_simples = (np.exp(log_return)-1)*100
                # change_pct = ((recent_price / old_price) - 1)*100
                heatmap_BROAD_data.loc[asset_BROAD,label] = log_return
            except:
                heatmap_BROAD_data.loc[asset_BROAD,label] = np.nan



    # Adicionar pesos
    heatmap_BROAD_data['Weight (%)'] = heatmap_BROAD_data.index.map(weights_BROAD)
    cols = ['Weight (%)'] + [c for c in heatmap_BROAD_data.columns if c != 'Weight (%)']
    heatmap_BROAD_data = heatmap_BROAD_data[cols]

    # Formatação
    cols_for_color = [c for c in heatmap_BROAD_data.columns if c != 'Weight (%)']
    for c in cols_for_color:
        heatmap_BROAD_data[c] = pd.to_numeric(heatmap_BROAD_data[c], errors='coerce')
    abs_max = np.abs(heatmap_BROAD_data[cols_for_color].values).max()

    # Exibir como DataFrame normal
    st.write("### Heatmap BROAD Currencies")
    st.dataframe(heatmap_BROAD_data.style
                 .background_gradient(cmap='RdBu',
                                      subset=cols_for_color,
                                      vmin=-abs_max,
                                      vmax=abs_max)
                 .format("{:+.2f}%", subset=cols_for_color)
                 .format("{:.1f}%", subset=['Weight (%)'])
                )
    
    ### CORRELAÇÃO DAS MOEDAS 

    assets_BROAD = ['USDEUR','USDCNY','USDCAD','USDMXN','USDJPY','USDGBP','USDKRW','USDINR','USDCHF','USDBRL','USDTWD','USDSGD','USDHKD','USDAUD','USDVND','USDMYR','USDTHB','USDILS','USDIDR','USDPHP','USDCLP','USDCOP','USDSAR','USDARS','USDRUB','USDSEK']

    df_Moedas = df_BROAD[assets_BROAD].copy()

    df_Moedas_log = np.log(df_Moedas).diff().dropna(how = 'any')

    corr_moedas = df_Moedas_log.corr()

    st.write("### Matriz de Correlação - BROAD Moedas")
    st.dataframe(
        corr_moedas.style
        .background_gradient(cmap="RdBu", vmin=-1, vmax=1)
        .format("{:.2f}")
    )
if __name__ == "__main__":
    main()