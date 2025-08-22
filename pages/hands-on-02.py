import streamlit as st

with st.container(horizontal=True, horizontal_alignment='center'):
    if st.button('< Voltar'):
        st.switch_page('pages/hands-on-01.py')
