import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

st.set_page_config(page_title="Projeto U1", layout="wide")

st.title("Projeto U1")
st.write("---")
st.subheader('1 - Questão do projeto.')
st.write('No arquivo deste experimento temos um arquivo .mat com:')
st.write('* fs: frequência de amostragem (taxa de amostragem)')
st.write('* t_sample: vetor com os instantes das amostras do sinal')
st.write('* x: amostras dos sinal')
st.write('O sinal x é composto por uma soma de 2 a 5 senos com frequências e fases diferentes. ' \
        'Seu objetivo é, por meio do sinal , estimar quantos senos compõem o sinal, quais suas ' \
        'frequências e respectivas fases.')
st.write("---")
st.subheader(
    '2 - A solução para nosso problema é utilizar uma Fast Fourier Transform (FFT) e analisar as fases com um Phase Spectrum.'
)

if "use_test" not in st.session_state:                                        # Flag inicial
    st.session_state.use_test = False

user_uploaded_file = st.file_uploader("Envie seu arquivo .mat", type=["mat"]) # Upload do arquivo
if user_uploaded_file is not None:                                            # Teste arquivo valido
    st.session_state.use_test = False
    max_size_mb = 5                                                           # tamanho máximo permitido
    if user_uploaded_file.size > max_size_mb * 1024 * 1024:
        st.error(f"O arquivo é muito grande! Máximo permitido: {max_size_mb} MB")
        user_uploaded_file = None                                             # ignora arquivo grande
elif st.button("Ou utilize o meu arquivo do projeto, clique aqui!"):          # Determinar Teste
    st.session_state.use_test = True

if st.session_state.use_test:                                                 # Determinar se é teste ou não
    uploaded_file = "pages/etc/sample.mat"
else:
    uploaded_file = user_uploaded_file

if uploaded_file is not None:
    try:                                                                 
        mat_data = loadmat(uploaded_file)                                # Carregar o .mat
        keys = [k for k in mat_data.keys() if not k.startswith("__")]    # Ignora metadados e armazena variaveis

        st.write("* Variáveis encontradas no arquivo:")
        x = st.selectbox("Selecione o vetor amostras dos sinal", keys)
        t_sample = st.selectbox("Selecione o vetor com os instantes das amostras do sinal", keys)
        fs = st.selectbox("Selecione o vetor frequência de amostragem (taxa de amostragem)", keys)
        
        t_sample = np.ravel(mat_data[t_sample])                          # formatar valores
        fs = np.ravel(mat_data[fs])                                      # ---
        signal = np.ravel(mat_data[x])                                   # ---

        if st.button("Analisar Sinal"):
            fft_vals = np.fft.rfft(signal)                               # * FFT do sinal (r para parte positiva)
            N = len(signal)                                              # N pontos
            T = 1/fs                                                     # passo de tempo T

            fft_freq = np.fft.rfftfreq(N, T)                             # Frequencia valores no tempo
            fft_freq = fft_freq[:]/1000                                  # Ajuste para plot
            fft_magnitude = np.abs(fft_vals)                             # Magnitude de espectro

            peaks, _ = find_peaks(fft_magnitude, height=0.05)            # Achando indices dos picos
            peak_freqs = fft_freq[peaks]                                 # Achando valores com indices
            peak_magnitudes = fft_magnitude[peaks]                       # ---

            df_fft = pd.DataFrame({                                      # Formatando dados em uma tabela
                "Frequência(Hz)": peak_freqs,                   
                "Magnitude": peak_magnitudes,
            })          
            df_fft.index = range(1, len(df_fft) + 1)                     # ---
                                                                         # * Espectro de Fase
            mask_phase = np.abs(fft_vals) > 1e-3                         # Só mantém se a amplitude for relevante
            phase = np.angle(fft_vals, deg=True)                         # Angulos em graus do sinal

            phase_peaks = phase[peaks]                                   # Formatando dados em uma tabela
            df_phase = pd.DataFrame({                                    # ---
                "Frequência(Hz)": peak_freqs,                   
                "Graus": phase_peaks,
            })
            df_phase.index = range(1, len(df_fft) + 1)
            st.divider()
            tab1, tab2, tab3 = st.tabs(["Sinal no tempo", "FFT", "Resultados"]) # Abas para gráficos
            with tab1:
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=t_sample, y=signal, mode="lines", name="Sinal"))
                fig_time.update_layout(title="Sinal no Tempo", xaxis_title="Tempo (s)", yaxis_title="Amplitude")
                st.plotly_chart(fig_time, use_container_width=True)

            with tab2:
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Bar(x=fft_freq[:N//2], y=fft_magnitude[:N//2], name="FFT"))
                fig_fft.update_layout(title="Espectro de Frequência", xaxis_title="Frequência (Hz)", yaxis_title="Magnitude")
                st.plotly_chart(fig_fft, use_container_width=True)
            
            with tab3:
                st.write(
                    "* Tabela com frequências e magnitudes dos senos encontrados usando 'find_peaks':", 
                    df_fft, 
                    f"Como podemos ver, encontramos {len(df_fft)} senos!"
                )

                st.write("* Podemos encontrar o Espectro de Fase com os angulos usando 'np.angle':")  # Achando fases do sinal

                fig_phase = go.Figure()                                                               # Plot da fase na frequência
                fig_phase.add_trace(go.Scatter(x=fft_freq[mask_phase], y=phase[mask_phase], name="Phase Spectrum"))
                fig_phase.update_layout(title="Espectro de Fase", xaxis_title="Frequência (Hz)", yaxis_title="Fases(graus)")
                st.plotly_chart(fig_phase, use_container_width=True)

                st.write(
                    "Encontramos como resultado final usando 'np.angle' os graus das fases!", 
                    df_phase
                )

                st.write("* Como um extra, podemos utilizar o python para ouvir nosso sinal! Aqui está ele em áudio:")
                fs_a = fs/1000                                                                        # Formatando frequência
                st.audio(signal, sample_rate=fs_a[0])                                                 # Sinal e frequência para o audio

            st.write("---")
            st.subheader("3 - Códigos executados.")                                  # Mostrar código
            tab1, tab2, tab3 = st.tabs(["Tratando dados em .mat", "Plot de gráficos", "Resultados"])
            with tab1:
                st.code(f"""
mat_data = loadmat({{uploaded_file}})                            # Carregar o .mat
keys = [k for k in mat_data.keys() if not k.startswith("__")]    # Ignora metadados e armazena variaveis

st.write("* Variáveis encontradas no arquivo:")                    # Mostrar e selecionar variaveis
x = st.selectbox("Selecione o vetor amostras dos sinal", keys)
t_sample = st.selectbox("Selecione o vetor com os instantes das amostras do sinal", keys)
fs = st.selectbox("Selecione o vetor frequência de amostragem (taxa de amostragem)", keys)
        
t_sample = np.ravel(mat_data[t_sample])                          # Formata em vetores pelas chaves
fs = np.ravel(mat_data[fs])                                      # ---
signal = np.ravel(mat_data[x])                                   # ---

if st.button("Analisar Sinal"):
    fft_vals = np.fft.rfft(signal)                               # * FFT do sinal (r para parte positiva)
    N = len(signal)                                              # N pontos
    T = 1/fs                                                     # passo de tempo T

    fft_freq = np.fft.rfftfreq(N, T)                             # Frequencia valores no tempo
    fft_freq = fft_freq[:]/1000                                  # Ajuste para plot
    fft_magnitude = np.abs(fft_vals)                             # Magnitude de espectro

    peaks, _ = find_peaks(fft_magnitude, height=0.05)            # Achando indices dos picos
    peak_freqs = fft_freq[peaks]                                 # Achando valores com indices
    peak_magnitudes = fft_magnitude[peaks]                       # ---

    df_fft = pd.DataFrame({{                                      # Formatando dados em uma tabela
        "Frequência(Hz)": peak_freqs,                   
        "Magnitude": peak_magnitudes,
    }})          
    df_fft.index = range(1, len(df_fft) + 1)                     # ---
                                                                 # * Espectro de Fase
    mask_phase = np.abs(fft_vals) > 1e-3                         # Só mantém se a amplitude for relevante
    phase = np.angle(fft_vals, deg=True)                         # Angulos em graus do sinal

    phase_peaks = phase[peaks]                                   # Formatando dados em uma tabela
    df_phase = pd.DataFrame({{                                   # ---
        "Frequência(Hz)": peak_freqs,                   
        "Graus": phase_peaks,
    }})
    df_phase.index = range(1, len(df_fft) + 1)
""", language="python")
            with tab2:
                st.code(f"""
if st.button("Analisar Sinal"):
    ...
    tab1, tab2, tab3 = st.tabs(["Sinal no tempo", "FFT", "Resultados"]) # Abas para gráficos e respostas

    with tab1:                                                          # Sinal no tempo com plotly
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_sample, y=signal, mode="lines", name="Sinal"))
        fig_time.update_layout(title="Sinal no Tempo", xaxis_title="Tempo (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_time, use_container_width=True)

    with tab2:                                                          # Fast Fourier Transform
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Bar(x=fft_freq[:N//2], y=fft_magnitude[:N//2], name="FFT"))
        fig_fft.update_layout(title="Espectro de Frequência", xaxis_title="Frequência (Hz)", yaxis_title="Magnitude")
        st.plotly_chart(fig_fft, use_container_width=True)
""", language="python")
            with tab3:
                st.code(f"""
    ...
    with tab3:
        st.write(
            "* Tabela com frequências e magnitudes dos senos encontrados usando 'find_peaks':", 
            df_fft, 
            f"Como podemos ver, encontramos {{len(df_fft)}} senos!"
        )

        st.write("* Podemos encontrar o Espectro de Fase com os angulos usando 'np.angle':")  # Achando fases do sinal

        fig_phase = go.Figure()                                                               # Plot da fase na frequência
        fig_phase.add_trace(go.Scatter(x=fft_freq[mask_phase], y=phase[mask_phase], name="Phase Spectrum"))
        fig_phase.update_layout(title="Espectro de Fase", xaxis_title="Frequência (Hz)", yaxis_title="Fases(graus)")
        st.plotly_chart(fig_phase, use_container_width=True)

        st.write(
            "Encontramos como resultado final usando 'np.angle' os graus das fases!", 
            df_phase
        )
                        
        st.write("* Como um extra, podemos utilizar o python para ouvir nosso sinal! Aqui está ele em áudio:")
        fs_a = fs/1000                                                                        # Formatando frequência
        st.audio(signal, sample_rate=fs_a[0])                                                 # Sinal e frequência para o audio
""", language="python")

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")

st.markdown('---')
with st.container(horizontal=True, horizontal_alignment='center'):       # Fim da pagina
    if st.button('< Voltar'):
        st.switch_page('pages/about-me.py')
    st.link_button('Pagina do projeto U1','https://github.com/vicentesousa/DCO1005/blob/main/h03_python.ipynb')
    if st.button('Projeto U2 >'):
        st.switch_page('pages/proj-02.py')