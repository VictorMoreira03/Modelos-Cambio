import streamlit as st

##---- PAGE SETUP ----##

DXY = st.Page(page= "pages/1_Moedas_DXY.py",title="DXY Currencies Analysis", default=True)

EME = st.Page(page= "pages/2_Moedas_EME.py",title="EME Currencies Analysis")

Broad = st.Page(page= "pages/3_Moedas_Broad.py",title="Broad Currencies Analysis")

Bolsas_Globais = st.Page(page= "pages/4_Bolsas.py",title="Global Equities Analysis",icon="📈")

Bonds_Globais = st.Page(page= "pages/5_Juros.py",title="Global Bonds Analysis",icon="📊")


##---- NAVIGATION SETUP ----##

# pg = st.navigation(pages=[project_1_page, project_2_page, project_3_page])


#---- NAVIGATION SETUP (com seções) ----##

pg = st.navigation(
    {
        "Moedas": [DXY,EME,Broad],
        "Bolsas": [Bolsas_Globais],
        "Juros": [Bonds_Globais],
    }
)

# --- Shared on All Pages --- ##
st.logo("assets/butialogo.png")

# st.sidebar.text("Fonte: Yahoo Finance")
# st.sidebar.text("Feito para uso interno")


pg.run()