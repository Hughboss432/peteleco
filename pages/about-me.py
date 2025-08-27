import streamlit as st

st.set_page_config(page_title='About Me', layout='wide')

with st.container(horizontal=True, horizontal_alignment='center'):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        c1, c2 = st.columns([2,6], border=True)
        with c1:
            st.image(
                'pages/etc/dev.jpeg',
                width=150,
            )
        with c2:
            st.markdown('• Victor Hugh')
            st.write('• Estudante de Ciências e Tecnologia')
            c1, c2 = st.columns([1,3])
            with c1:
                st.link_button('🔗 LinkedIn','https://www.linkedin.com/in/victor-hugh-03031b23b')
            with c2:
                st.link_button('💻 GitHub','https://github.com/Hughboss432')
        
        

st.markdown('---')

with st.container(horizontal=True, horizontal_alignment='center'):
    # bottom buttons
    if st.button('< Voltar'):
        st.switch_page('pages/home.py')
    if st.button('Projeto U1 >'):
        st.switch_page('pages/proj-01.py')