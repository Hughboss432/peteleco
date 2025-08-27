import streamlit as st

st.set_page_config(page_title='meu app', layout='wide')

pg = st.navigation([
    st.Page('pages/home.py', title='Inicio'),
    st.Page('pages/about-me.py', title='Sobre mim'),
    st.Page('pages/proj-01.py', title='Projeto U1'),
])

pg.run()