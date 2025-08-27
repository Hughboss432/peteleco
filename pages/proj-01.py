import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat

st.set_page_config(page_title="Projeto U1", layout="wide")

st.title("Projeto U1")

st.subheader('1 - Questão do projeto a ser resolvida.')
st.write('No arquivo deste experimento temos um arquivo .mat com:')
st.write('* fs: frequência de amostragem (taxa de amostragem)')
st.write('* t_sample: vetor com os instantes das amostras do sinal')
st.write('* x: amostras dos sinal')
st.write('O sinal x é composto por uma soma de 2 a 5 senos com frequências e fases diferentes. ' \
        'Seu objetivo é, por meio do sinal , estimar quantos senos compõem o sinal, quais suas ' \
        'frequências e respectivas fases.')

st.subheader('2 - A solução para nosso problema é utilizar uma Fast Fourier Trasform (FFT).')

uploaded_file = st.file_uploader("Envie um arquivo .mat", type=["mat"])  # Upload do arquivo

if uploaded_file is not None:
    try:                                                                 
        mat_data = loadmat(uploaded_file)                                # Carregar o .mat
        keys = [k for k in mat_data.keys() if not k.startswith("__")]    # Ignora metadados e armazena variaveis

        st.write("Variáveis encontradas no arquivo:")
        x = st.selectbox("Selecione o vetor amostras dos sinal", keys)
        t_sample = st.selectbox("Selecione o vetor com os instantes das amostras do sinal", keys)
        fs = st.selectbox("Selecione o vetor frequência de amostragem (taxa de amostragem)", keys)
        
        t_sample = np.ravel(mat_data[t_sample])                          # só pega o vetor quando selecionado 
        fs = np.ravel(mat_data[fs])                                      # ---
        signal = np.ravel(mat_data[x])                                   # ---

        if st.button("Executar FFT"):
            fft_vals = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(t_sample), 1/fs)
            fft_magnitude = np.abs(fft_vals) / len(signal)

            st.subheader("Códigos executados:")                            # Mostrar código
            tab1, tab2 = st.tabs(["Importando o arquivo .mat", "Plot de graficos"])
            with tab1:
                st.code(f"""
mat_data = loadmat('{uploaded_file.name}')                       # Carregar o .mat
keys = [k for k in mat_data.keys() if not k.startswith("__")]    # Ignora metadados e armazena variaveis

st.write("Variáveis encontradas no arquivo:")                    # Mostrar e selecionar variaveis
x = st.selectbox("Selecione o vetor amostras dos sinal", keys)
t_sample = st.selectbox("Selecione o vetor com os instantes das amostras do sinal", keys)
fs = st.selectbox("Selecione o vetor frequência de amostragem (taxa de amostragem)", keys)
        
t_sample = np.ravel(mat_data[t_sample])                          # Formata em vetor matlab para np
fs = np.ravel(mat_data[fs])                                      # ---
signal = np.ravel(mat_data[x])                                   # ---

if st.button("Executar FFT"):                                    # Executar FFT
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(t_sample), 1/fs)
    fft_magnitude = np.abs(fft_vals) / len(signal)
""", language="python")
            with tab2:
                st.code(f"""
tab1, tab2 = st.tabs(["Sinal no tempo", "FFT"])              # Abas para gráficos

with tab1:                                                   # Sinal no tempo com plotly
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t_sample, y=signal, mode="lines", name="Sinal"))
    fig_time.update_layout(title="Sinal no Tempo", xaxis_title="Tempo (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_time, use_container_width=True)

with tab2:                                                   # Fast Fourier Transform
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Bar(x=fft_freq[:len(fft_freq)//2], 
                             y=fft_magnitude[:len(fft_magnitude)//2], 
                             name="FFT"))
    fig_fft.update_layout(title="Espectro de Frequência", xaxis_title="Frequência (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig_fft, use_container_width=True)
""", language="python")

            tab1, tab2 = st.tabs(["Sinal no tempo", "FFT"])              # Abas para gráficos

            with tab1:
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=t_sample, y=signal, mode="lines", name="Sinal"))
                fig_time.update_layout(title="Sinal no Tempo", xaxis_title="Tempo (s)", yaxis_title="Amplitude")
                st.plotly_chart(fig_time, use_container_width=True)

            with tab2:
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Bar(x=fft_freq[:len(fft_freq)//2], 
                                         y=fft_magnitude[:len(fft_magnitude)//2], 
                                         name="FFT"))
                fig_fft.update_layout(title="Espectro de Frequência", xaxis_title="Frequência (Hz)", yaxis_title="Magnitude")
                st.plotly_chart(fig_fft, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")

st.markdown('---')
with st.container(horizontal=True, horizontal_alignment='center'):       # Fim da pagina
    if st.button('< Voltar'):
        st.switch_page('pages/about-me.py')