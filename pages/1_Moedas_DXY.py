
import streamlit as st
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 

st.set_page_config(page_title="DXY Currencies Analysis")



def main(): 
   

    tickers_DXY = { 
    'DXY': 'DX-Y.NYB',
    'USDEUR': 'USDEUR=X',
    'USDGBP': 'USDGBP=X',
    'USDSEK': 'USDSEK=X',
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'USDCAD': 'USDCAD=X'
    } 

    weights_DXY = {
        'USDEUR': 57.6,
        'USDJPY': 13.6,
        'USDGBP': 11.9,
        'USDCAD': 9.1,
        'USDSEK': 4.2,
        'USDCHF': 3.6,
        'DXY': 100
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

    data_DXY = yf.download(list(tickers_DXY.values()),
                            period = '30y',
                            interval = '1d')['Close']

    df_DXY = pd.DataFrame()

    for asset_DXY, ticker_DXY in tickers_DXY.items(): 
        df_DXY[asset_DXY] = data_DXY[ticker_DXY]

    df_DXY1 = df_DXY[['DXY']].ffill()

    df_DXY1['DXY_200d'] = df_DXY1['DXY'].rolling(window=200).mean()
    df_DXY1['DXY_50d'] = df_DXY1['DXY'].rolling(window=50).mean()

    df_plot_DXY = df_DXY1[['DXY','DXY_200d','DXY_50d']]

    fig_dxy = go.Figure()

    fig_dxy.add_trace(go.Scatter(
        x=df_plot_DXY.index,
        y=df_plot_DXY["DXY"],
        mode="lines",
        name="DXY",                      # Legenda na chart
        line=dict(color="black")          # Define a cor da linha
    ))

    fig_dxy.add_trace(go.Scatter(
        x=df_plot_DXY.index,
        y=df_plot_DXY["DXY_200d"],
        mode="lines",
        name="200d Média",
        line=dict(color="red")           # Define a cor para a média móvel
    ))

    fig_dxy.add_trace(go.Scatter(
        x=df_plot_DXY.index,
        y=df_plot_DXY["DXY_50d"],
        mode="lines",
        name="50d Média",
        line=dict(color="blue")           # Define a cor para a média móvel
    ))

    fig_dxy.update_layout(
        title="Performance do DXY (10 anos)",
        yaxis=dict(range=[70, 130]),
        xaxis=dict(range=[
            df_plot_DXY.index[0], 
            df_plot_DXY.index[-1] + pd.Timedelta("40d")
        ])
    )

    st.write("### DXY Index")
    st.plotly_chart(fig_dxy, use_container_width = True)


    assets_DXY = ['USDEUR', 'USDGBP', 'USDSEK', 'USDJPY', 'USDCHF', 'USDCAD', 'DXY']
    heatmap_DXY_data = pd.DataFrame( index = assets_DXY, columns = windows.keys())

    for asset_DXY in assets_DXY: 
        for label, w in windows.items():
            #Pega o valor mais recente e compara com w dias
            try: 
                recent_price = df_DXY[asset_DXY].iloc[-1]
                old_price = df_DXY[asset_DXY].iloc[-(1+w)]
                log_return = (np.log(recent_price) - np.log(old_price))*100
                heatmap_DXY_data.loc[asset_DXY,label] = log_return
            except:
                heatmap_DXY_data.loc[asset_DXY,label] = np.nan


    # Adicionar pesos
    heatmap_DXY_data['Weight (%)'] = heatmap_DXY_data.index.map(weights_DXY)
    cols = ['Weight (%)'] + [c for c in heatmap_DXY_data.columns if c != 'Weight (%)']
    heatmap_DXY_data = heatmap_DXY_data[cols]

    # Formatação
    cols_for_color = [c for c in heatmap_DXY_data.columns if c != 'Weight (%)']
    for c in cols_for_color:
        heatmap_DXY_data[c] = pd.to_numeric(heatmap_DXY_data[c], errors='coerce')
    abs_max = np.abs(heatmap_DXY_data[cols_for_color].values).max()

    # Exibir como DataFrame normal
    st.write("### Heatmap DXY Currencies")
    st.dataframe(heatmap_DXY_data.style
                 .set_properties(**{'text-align': 'center'})  # alinha o conteúdo das células
                 .set_table_styles([
                    {'selector': 'th', 'props': [('text-align','center')]},  # cabeçalhos
                    {'selector': 'td', 'props': [('text-align','center')]}   # células
                 ])
                 .background_gradient(cmap='RdBu', 
                                     subset=cols_for_color,
                                     vmin=-abs_max, 
                                     vmax=abs_max)
                 .format("{:+.2f}%", subset=cols_for_color)
                 .format("{:.1f}%", subset=['Weight (%)'])
                )
if __name__ == "__main__":
    main()