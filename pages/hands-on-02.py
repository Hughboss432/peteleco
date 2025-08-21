import streamlit as st

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button('Voltar'):
        st.switch_page('pages/hands-on-01.py')
