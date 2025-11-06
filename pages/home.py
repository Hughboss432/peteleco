import streamlit as st

st.set_page_config(page_title='Projetos PETELECO', layout='wide')

with st.container(horizontal=True, horizontal_alignment='center', border=True):
    col1, col2, col3= st.columns([3,1,1])
    with col1:
        col1, col2= st.columns([2,8])
        with col1:
            st.image('https://www.ufrn.br/resources/documentos/identidadevisual/brasao/brasao_gradiente.png', width=100)
        with col2:
            st.markdown('• Universidade Federal do Rio Grande do Norte')
            st.markdown('• Graduação em Ciências e Tecnologia')
            st.markdown('• DCO1005 - Princípios de Telecomunicações')

st.markdown('---')
st.markdown(
    '''
    <div style="text-align: center; margin-top: 50px; margin-bottom: 50px;">
        <h1 style="font-size: 48px; color: #1f77b4;">
            📡​ Projetos PETELECO
        </h1>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('---')

# buttons for more info
with st.container(horizontal=True, horizontal_alignment='center'):
    if st.button('Sobre mim >'):
        st.switch_page('pages/about-me.py')