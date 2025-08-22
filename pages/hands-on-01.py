import streamlit as st

with st.container(horizontal=True, horizontal_alignment='center'):
    if st.button('< Voltar'):
        st.switch_page('pages/about-me.py')
    if st.button('Hands-on-02 >'):
        st.switch_page('pages/hands-on-02.py')