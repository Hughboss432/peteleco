import streamlit as st

st.set_page_config(page_title='meu app', layout='wide')

pg = st.navigation([
    st.Page('pages/home.py', title='Inicio'),
    st.Page('pages/about-me.py', title='Sobre mim'),
    st.Page('pages/hands-on-01.py', title='Hands-on-01'),
    st.Page('pages/hands-on-02.py', title='Hands-on-02'),
])

pg.run()