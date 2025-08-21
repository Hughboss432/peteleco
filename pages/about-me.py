import streamlit as st

st.set_page_config(page_title='About Me', layout='wide')

col1, col2, col3 = st.columns([1,2,1])
with col2:
    c1, c2 = st.columns([1,1])
    with c1:
        st.image(
            'pages/etc/dev.jpg',
            width=150
        )
    with c2:
        st.markdown('Victor Hugh')
        st.write('Estudante de Ciências e Tecnologia')
    
    c1, c2 = st.columns([1,1])
    with c1:
        st.link_button('🔗 LinkedIn','https://www.linkedin.com/in/victor-hugh-03031b23b')
    with c2:
        st.link_button('💻 GitHub','https://github.com/Hughboss432')

st.markdown('---')
# bottom buttons
col1, col2, col3 = st.columns([1,2,1])
with col1:
    if st.button('Voltar'):
        st.switch_page('pages/home.py')
with col3:
    if st.button('Hands-on-01'):
        st.switch_page('pages/hands-on-01.py')