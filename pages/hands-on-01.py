import streamlit as st

col1, col2, col3 = st.columns([1,2,1])
with col1:
    if st.button('Voltar'):
        st.switch_page('pages/about-me.py')
with col3:
    if st.button('Hands-on-02'):
        st.switch_page('pages/hands-on-02.py')