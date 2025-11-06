import streamlit as st

st.set_page_config(page_title='Loading...', layout='wide')

pg = st.navigation([
    st.Page('pages/home.py', title='Inicio'),
    st.Page('pages/about-me.py', title='Sobre mim'),
    st.Page('pages/proj-01.py', title='Projeto U1'),
    st.Page('pages/proj-02.py', title='Projeto U2'),
])

pg.run()