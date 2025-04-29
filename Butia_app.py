import streamlit as st

##---- PAGE SETUP ----##

moedas_DXY_page = st.Page(
    page= "pages/1_Moedas_DXY.py",
    title="DXY Currencies Analysis",
    icon="💱",
    default= True,
)
moedas_EME_page = st.Page(
    page= "pages/2_Moedas_EME.py",
    title="EME Currencies Analysis",

)
moedas_BROAD_page = st.Page(
    page= "pages/3_Moedas_Broad.py",
    title="Broad Currencies Analysis",
)
project_2_page = st.Page(
    page= "pages/2_Bolsas.py",
    title="Rates Analysis",
    icon="📈",
)
project_3_page = st.Page(
    page= "pages/3_Juros.py",
    title="Sectors Analysis",
    icon="📊",
)


##---- NAVIGATION SETUP ----##

# pg = st.navigation(pages=[project_1_page, project_2_page, project_3_page])


#---- NAVIGATION SETUP (com seções) ----##

pg = st.navigation(
    {
        "Moedas": [moedas_DXY_page,moedas_EME_page,moedas_BROAD_page],
        "Bolsas": [project_2_page],
        "Juros": [project_3_page],
    }
)
pg.run()