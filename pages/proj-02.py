import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.io import loadmat
from scipy import fftpack

st.set_page_config(page_title="Projeto U2", layout="wide")

st.title("Projeto U2")

st.markdown('---')

st.subheader('1 - Primeira questão do projeto.')
st.write('Mude o código (exemplo sobre modulação AM-DSB) em Python para expressar os três casos clássicos de modulação: ')
st.write('* (i) 100% de modulação;')
st.write('* (ii) submodulação;')
st.write('* (iii) sobremodulação;')
st.write('Explique a diferença entres eles e evidencie nos gráficos suas características.')

tab1, tab2, tab3 = st.tabs(["Código exemplo","Resposta para o problema", "Plot de gráficos"])
with tab1:
    st.code(f"""
# Importa bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Parâmetros do sinal
Ac = 2                                                         # Amplitude da portadora
Mu = 0.7                                                       # Índice de modulação
fc = 25000                                                     # Frequência da portadora Hz
fm = 2000
N = 1000
Ts = 1e-6                                                      # Tempo de amostragem pequeno (modelar sinal contínuo)
t = np.arange(N)*Ts
s = Ac*(1+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t)

# Cálculo da FFT de AM-DSB
lfft = 30                                                      # Número pontos da fft
k = np.arange(-lfft,lfft)                                      # Vetor de frequências
S_f = 2.0*np.abs((fftpack.fft(s)))/N                           # Cálculo da FFT
Ns = len(s)                                                    # Comprimento do sinal modulado
Nk = len(k)                                                    # Comprimento do sinal em frequência

# A fft em 30 pontos (para melhor visualização)
S_f_new = np.zeros(Nk)                                         # Inicialização do vetor da frequência
fsampling = 1/Ts                                               # Taxa de amostragem
freq = (fsampling/Ns)*k                                        # Eixo de frequências
for i in range(Nk):
    kk = k[i]
    if kk>=0:
        S_f_new[i] = S_f[kk]
    else :
        S_f_new[i] = S_f[Ns+kk]

# Gráfico do AM-DSB no tempo
plt.figure(1,[10,7])
plt.subplot(211)
plt.plot(t,s)
plt.title("Sinal AM no tempo - AM-DSB (padrão)")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")

# Gráfico do AM-DSB na frequência
plt.subplot(212)
plt.title("AM-DSB na frequência")
plt.xlabel("Frequência [kHz]")
plt.ylabel("Magnitude")
plt.stem(freq/1e3,S_f_new)
plt.tight_layout()
plt.show()
    """, language="python")

with tab2:
    st.write('Para os três casos, podemos obter os resultados alterando o valor da amplitude da portadora, já que não podemos alterar a amplitude do sinal modulador. Para cada caso, em ordem, temos:')
    st.write('* ma = 1 (modulação de 100%), \n* ma < 1 (submodulação), \n* ma > 1 (sobremodulação).')
    st.write('As alterações no código foram feitas para criar uma função que recebe o valor da amplitude da portadora, além de algumas melhorias nos gráficos de plotagem com o Plotly.')
    st.code(f"""
# Importa bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import plotly.graph_objects as go

def mod_am_dsb(Ac,Am):
  # Parâmetros do sinal
  Ac = Ac                                                        # Amplitude da portadora
  Am = Am                                                        # Amplitude do sinal
  Mu = Am/Ac                                                     # Índice de modulação

  fc = 25000                                                     # Frequência da portadora Hz
  fm = 2000
  N = 1000
  Ts = 1e-6                                                      # Tempo de amostragem pequeno (modelar sinal contínuo)
  t = np.arange(N)*Ts
  s = Ac*(1+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t)

  # Cálculo da FFT de AM-DSB
  lfft = 30                                                      # Número pontos da fft
  k = np.arange(-lfft,lfft)                                      # Vetor de frequências
  S_f = 2.0*np.abs((fftpack.fft(s)))/N                           # Cálculo da FFT
  Ns = len(s)                                                    # Comprimento do sinal modulado
  Nk = len(k)                                                    # Comprimento do sinal em frequência

  # A fft em 30 pontos (para melhor visualização)
  S_f_new = np.zeros(Nk)                                         # Inicialização do vetor da frequência
  fsampling = 1/Ts                                               # Taxa de amostragem
  freq = (fsampling/Ns)*k                                        # Eixo de frequências
  for i in range(Nk):
      kk = k[i]
      if kk>=0:
          S_f_new[i] = S_f[kk]
      else :
          S_f_new[i] = S_f[Ns+kk]

  # Gráfico do AM-DSB no tempo usando Plotly
  fig_time = go.Figure()
  fig_time.add_trace(go.Scatter(x=t, y=s, mode='lines'))
  fig_time.update_layout(title=f"Sinal AM no tempo - AM-DSB (padrão)",
                         xaxis_title="Tempo [s]",
                         yaxis_title="Amplitude")
  fig_time.show()

  # Gráfico do AM-DSB na frequência usando Plotly
  fig_freq = go.Figure()
  fig_freq.add_trace(go.Bar(x=freq/1e3, y=S_f_new))
  fig_freq.update_layout(title=f"AM-DSB na frequência",
                         xaxis_title="Frequência [kHz]",
                         yaxis_title="Magnitude")
  fig_freq.show()
    """, language="python")
with tab3:
    def mod_am_dsb(Ac,Am):                                             # Função resolução
        # Parâmetros do sinal
        Ac = Ac                                                        # Amplitude da portadora
        Am = Am                                                        # Amplitude do sinal
        Mu = Am/Ac                                                     # Índice de modulação

        fc = 25000                                                     # Frequência da portadora Hz
        fm = 2000
        N = 1000
        Ts = 1e-6                                                      # Tempo de amostragem pequeno (modelar sinal contínuo)
        t = np.arange(N)*Ts
        s = Ac*(1+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t)

        # Cálculo da FFT de AM-DSB
        lfft = 30                                                      # Número pontos da fft
        k = np.arange(-lfft,lfft)                                      # Vetor de frequências
        S_f = 2.0*np.abs((fftpack.fft(s)))/N                           # Cálculo da FFT
        Ns = len(s)                                                    # Comprimento do sinal modulado
        Nk = len(k)                                                    # Comprimento do sinal em frequência

        # A fft em 30 pontos (para melhor visualização)
        S_f_new = np.zeros(Nk)                                         # Inicialização do vetor da frequência
        fsampling = 1/Ts                                               # Taxa de amostragem
        freq = (fsampling/Ns)*k                                        # Eixo de frequências
        for i in range(Nk):
            kk = k[i]
            if kk>=0:
                S_f_new[i] = S_f[kk]
            else :
                S_f_new[i] = S_f[Ns+kk]

        # Gráfico do AM-DSB no tempo usando Plotly
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t, y=s, mode='lines'))
        fig_time.update_layout(title=f"Sinal AM no tempo - AM-DSB (padrão)",
                                xaxis_title="Tempo [s]",
                                yaxis_title="Amplitude")

        # Gráfico do AM-DSB na frequência usando Plotly
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(x=freq/1e3, y=S_f_new))
        fig_freq.update_layout(title=f"AM-DSB na frequência",
                                xaxis_title="Frequência [kHz]",
                                yaxis_title="Magnitude")
        
        return fig_time, fig_freq

    st.write('* Aplicando 100% de modulação alterando a amplitude da portadora(Ac):')
    u_mod_t, u_mod_f = mod_am_dsb(Ac=1.4,Am=1.4)
    st.plotly_chart(u_mod_t)
    st.plotly_chart(u_mod_f)
    st.write('* Aplicando submodulação alterando a amplitude da portadora(Ac):')
    sub_mod_t, sub_mod_f = mod_am_dsb(Ac=2.8,Am=1.4)
    st.plotly_chart(sub_mod_t)
    st.plotly_chart(sub_mod_f)
    st.write('* Aplicando sobremodulação alterando a amplitude da portadora(Ac):')
    sob_mod_t, sob_mod_f = mod_am_dsb(Ac=0.7,Am=1.4)
    st.plotly_chart(sob_mod_t)
    st.plotly_chart(sob_mod_f)

st.markdown('---')

st.subheader('2 - Segunda questão do projeto.')
st.write('Escreva um script em python que calcule o erro médio quadrático entre a envoltória ideal e a envoltória recuperada para os seguintes valores de τ: ')
st.write('* (i) $τ = 1 * 10^{-1}$;')
st.write('* (ii) $τ = 2 * 10^{-4}$;')
st.write('* (iii) $τ = 4 * 10^{-3}$;')
st.write('Disserte sobre os três valores de e o valor do erro. Quais os fatores causadores do erro?')

tab1, tab2, tab3 = st.tabs(["Código exemplo","Resposta para o problema", "Plot de gráficos"])
with tab1:
    st.code(f"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

tau = 1e-4                                                      # Constante de tempo do detector de envelope
Ts=1e-6                                                         # Definição do período
t = np.arange(1000)*Ts                                          # Definição do vetor tempo
fc = 10000                                                      # Frequência da portadora.
fm = 1500                                                       # Frequência do sinal
Mu = 0.6                                                        # Índice de modulaçao.
Ac = 1.0
x_AMo = Ac*(1.0+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t);  # Onda Modulada AM-DSB

x_envIdeal = np.abs(Ac*(1.0+Mu*np.cos(2*np.pi*fm*t)))           # Envoltória ideal

# Detector de envoltória
x_AM = x_AMo*(x_AMo>0)                                          # Efeito do diodo (semiciclo positivo)
x_env = np.zeros(len(x_AM))
Ns = len(x_AM)
out = -1
for i in range(Ns):
    inp = x_AM[i]
    if inp>=out:
        out = inp            # Caso 1: x_am(t) > Vc(t) (carga do capacitor)
    else:
        out *= (1-Ts/tau)    # Caso 2: x_am(t) < Vc(t) (descarga do capacitor)
    x_env[i] = out

# gráfico composto
plt.figure(1,[10,7])
plt.title("Detecção de envoltória")
plt.ylabel("Amplitude")
plt.xlabel("Tempo(ms)")
envoltoria_ideal = plt.plot(t*1000,x_envIdeal)
sinal_transmitido = plt.plot(t*1000,x_AM)
detector_de_saida = plt.plot(t*1000,x_env)
plt.grid()
plt.legend(["Envoltória ideal","Onda AM retificada","Envoltória recuperada"])
plt.show()

## Gráficos com a função plt.plot()
plt.figure(1,[10,7])
plt.subplot(311)
plt.plot(t*1000,x_envIdeal)
plt.title("Envoltória ideal")
plt.ylabel("Amplitude")

plt.figure(1,[10,7])
plt.subplot(312)
plt.plot(t*1000,x_AM)
plt.title("Onda AM retificada")
plt.ylabel("Amplitude")

plt.figure(1,[10,7])
plt.subplot(313)
plt.plot(t*1000,x_env)
plt.title("Envoltória recuperada")
plt.ylabel("Amplitude")
plt.xlabel("Tempo(ms)")

plt.subplots_adjust(hspace=0.3) # Ajustando espaço entre subplots

plt.show()
    """, language="python")

with tab2:
    st.write('As principais modificações no código foram: ')
    st.write(
        '\nCalcular o erro quadrático médio com a média de $ (x_{\t{envIdeal}} - x_{\t{env}})^2 $; ' \
        'Alterar os gráficos para utilizar o Plotly.' \
        'O principal ponto de ajustar o valor de τ está relacionado à descarga e à carga do capacitor. ' \
        'Dependendo do valor escolhido, é possível recuperar com maior precisão a envoltória do sinal. No entanto, ' \
        'valores elevados podem levar a erros, pois o capacitor pode descarregar-se muito rapidamente (resultando em perda de informações) ' \
        'ou descarregar-se muito lentamente (resultando em retrabalho).')
    st.write(
        'Nesse caso, o valor de τ que mais se aproxima de recuperar os valores do nosso sinal é $ τ = 2 * 10^{-4} $, com o menor ' \
        'erro quadrático médio de $ 5.37*10^{-2} $.'
    )
    st.code('''
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def deteccao_envelope(tau):
  tau = tau                                                       # Constante de tempo do detector de envelope
  Ts=1e-6                                                         # Definição do período
  t = np.arange(1000)*Ts                                          # Definição do vetor tempo
  fc = 10000                                                      # Frequência da portadora.
  fm = 1500                                                       # Frequência do sinal
  Mu = 0.6                                                        # Índice de modulaçao.
  Ac = 1.0
  x_AMo = Ac*(1.0+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t);  # Onda Modulada AM-DSB

  x_envIdeal = np.abs(Ac*(1.0+Mu*np.cos(2*np.pi*fm*t)))           # Envoltória ideal

  # Detector de envoltória
  x_AM = x_AMo*(x_AMo>0)                                          # Efeito do diodo (semiciclo positivo)
  x_env = np.zeros(len(x_AM))
  Ns = len(x_AM)
  out = -1
  for i in range(Ns):
      inp = x_AM[i]
      if inp>=out:
          out = inp            # Caso 1: x_am(t) > Vc(t) (carga do capacitor)
      else:
          out *= (1-Ts/tau)    # Caso 2: x_am(t) < Vc(t) (descarga do capacitor)
      x_env[i] = out

  mse = np.mean((x_envIdeal - x_env) ** 2)                        # erro quadratico

  fig1 = go.Figure()                                              # plot composto
  fig1.add_trace(go.Scatter(x=t*1000, y=x_envIdeal, mode='lines', name='Envoltória ideal'))
  fig1.add_trace(go.Scatter(x=t*1000, y=x_AM, mode='lines', name='Onda AM retificada'))
  fig1.add_trace(go.Scatter(x=t*1000, y=x_env, mode='lines', name='Envoltória recuperada'))
  fig1.update_layout(title=f"Detecção de envoltória (τ = {{tau:.2e}})",
                      xaxis_title="Tempo (ms)",
                      yaxis_title="Amplitude")

  # Definindo subplots
  fig_subplots = make_subplots(rows=3, cols=1, subplot_titles=("Envoltória ideal", "Onda AM retificada", "Envoltória recuperada"))
  # Envoltoria ideal
  fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_envIdeal, mode='lines', name="Envoltória ideal"), row=1, col=1)
  fig_subplots.update_yaxes(title_text="Amplitude", row=1, col=1)
  # Onda AM retificada
  fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_AM, mode='lines', name="Onda AM retificada"), row=2, col=1)
  fig_subplots.update_yaxes(title_text="Amplitude", row=2, col=1)
  # Envoltoria recuperada
  fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_env, mode='lines', name="Envoltória recuperada"), row=3, col=1)
  fig_subplots.update_yaxes(title_text="Amplitude", row=3, col=1)
  fig_subplots.update_xaxes(title_text="Tempo (ms)", row=3, col=1)

  return fig1, fig_subplots, mse
    ''', language='python')

    st.write('Códigos para a média quadratica: ')
    st.code('''
x_envIdeal, x_env = deteccao_envelope(tau)
mse = np.mean((x_envIdeal - x_env) ** 2)
    ''', language='python')

with tab3:
    def det_env(tau):
        tau = tau                                                       # Constante de tempo do detector de envelope
        Ts=1e-6                                                         # Definição do período
        t = np.arange(1000)*Ts                                          # Definição do vetor tempo
        fc = 10000                                                      # Frequência da portadora.
        fm = 1500                                                       # Frequência do sinal
        Mu = 0.6                                                        # Índice de modulaçao.
        Ac = 1.0
        x_AMo = Ac*(1.0+Mu*np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t);  # Onda Modulada AM-DSB

        x_envIdeal = np.abs(Ac*(1.0+Mu*np.cos(2*np.pi*fm*t)))           # Envoltória ideal

        # Detector de envoltória
        x_AM = x_AMo*(x_AMo>0)                                          # Efeito do diodo (semiciclo positivo)
        x_env = np.zeros(len(x_AM))
        Ns = len(x_AM)
        out = -1
        for i in range(Ns):
            inp = x_AM[i]
            if inp>=out:
                out = inp            # Caso 1: x_am(t) > Vc(t) (carga do capacitor)
            else:
                out *= (1-Ts/tau)    # Caso 2: x_am(t) < Vc(t) (descarga do capacitor)
            x_env[i] = out

        mse = np.mean((x_envIdeal - x_env) ** 2)                        # erro quadratico

        fig1 = go.Figure()                                              # plot composto
        fig1.add_trace(go.Scatter(x=t*1000, y=x_envIdeal, mode='lines', name='Envoltória ideal'))
        fig1.add_trace(go.Scatter(x=t*1000, y=x_AM, mode='lines', name='Onda AM retificada'))
        fig1.add_trace(go.Scatter(x=t*1000, y=x_env, mode='lines', name='Envoltória recuperada'))
        fig1.update_layout(title=f"Detecção de envoltória (τ = {tau:.2e})",
                            xaxis_title="Tempo (ms)",
                            yaxis_title="Amplitude")

        # Definindo subplots
        fig_subplots = make_subplots(rows=3, cols=1, subplot_titles=("Envoltória ideal", "Onda AM retificada", "Envoltória recuperada"))
        # Envoltoria ideal
        fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_envIdeal, mode='lines', name="Envoltória ideal"), row=1, col=1)
        fig_subplots.update_yaxes(title_text="Amplitude", row=1, col=1)
        # Onda AM retificada
        fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_AM, mode='lines', name="Onda AM retificada"), row=2, col=1)
        fig_subplots.update_yaxes(title_text="Amplitude", row=2, col=1)
        # Envoltoria recuperada
        fig_subplots.add_trace(go.Scatter(x=t*1000, y=x_env, mode='lines', name="Envoltória recuperada"), row=3, col=1)
        fig_subplots.update_yaxes(title_text="Amplitude", row=3, col=1)
        fig_subplots.update_xaxes(title_text="Tempo (ms)", row=3, col=1)

        return fig1, fig_subplots, mse
                    

    st.write('* Aplicando erro médio quadrático para τ = 1e-1: ')
    t1_p1, t1_p2, mse1 = det_env(1e-1)
    st.write(f'Erro médio quadrático para τ = {1e-1}: {mse1:.2e}.')
    st.plotly_chart(t1_p1)
    st.plotly_chart(t1_p2)
    st.write('* Aplicando erro médio quadrático para τ = 2e-4:')
    t2_p1, t2_p2, mse2 = det_env(2e-4)
    st.write(f'Erro médio quadrático para τ = {2e-4}: {mse2:.2e}.')
    st.plotly_chart(t2_p1)
    st.plotly_chart(t2_p2)
    st.write('* Aplicando erro médio quadrático para τ = 4e-3:')
    t3_p1, t3_p2, mse3 = det_env(4e-3)
    st.write(f'Erro médio quadrático para τ = {4e-3}: {mse3:.2e}.')
    st.plotly_chart(t3_p1)
    st.plotly_chart(t3_p2)

st.markdown('---')

st.subheader(
    'colocar proxima questão'
)

st.markdown('---')
with st.container(horizontal=True, horizontal_alignment='center'):       # Fim da pagina
    if st.button('< Voltar'):
        st.switch_page('pages/proj-01.py')
    st.link_button('Pagina do projeto U2','https://github.com/vicentesousa/DCO1005/blob/main/h05.ipynb')
    #if st.button('Projeto U3 >'):
        #st.switch_page('pages/proj-03.py')