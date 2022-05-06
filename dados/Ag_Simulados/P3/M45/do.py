import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
import concurrent.futures
from scipy.stats import linregress
from scipy.interpolate import interp1d

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

nome = 'M45'
arquivo = 'm45.txt'
modulo_teorico = 5*np.log10(136) - 5
idade_teorica = 8.1

def global_var(x):
    global aglomerado, isocronas, E, idades, XAglo, YAglo, AGLO
    E = 0.03
    aglomerado =  pd.read_csv(x,comment = '#', skiprows = 51, header = None, usecols = [1,2], names = ['V','B-V'], delim_whitespace = True)
    isocronas = pd.read_csv('../../../Isocronas/isocro.csv', header = 0)
    idades = np.unique(isocronas['log(Age)'])
    XAglo = aglomerado['B-V']
    YAglo = aglomerado['V']
    AGLO = np.vstack((XAglo,YAglo)).T
global_var(arquivo)


def linear_func(p, x):
    m, c = p
    return m*x + c
def distancia(a,b,b0,E):
    return 10**((a*E + (b-b0) - E*3.1 + 5 )/5)
def jpt(X,Y):
    array = (X - Y)
    array = np.sum(array**2, axis = 1)
    A, B = np.argpartition(array, 1)[0:2]
    return A,B
def frayn(X1,X2,X3):
  h = 2*((X3[0]-X1[0])*(X2[1]-X1[1]) - (X3[1]-X1[1])*(X2[0]-X1[0]))/np.sqrt(dist(X1,X2))
  return h**2
def dist(X,Y):
  distancia = 0
  for j in range(len(X)):
    distancia+= (X[j] - Y[j])**2
  return distancia
def chi_to_age(IDADE):
  data = isocronas[isocronas['log(Age)'] == IDADE]
  Yiso = np.zeros((len(modulo_distancia), len(data)))
  for k in range(len(modulo_distancia)):
    Yiso[k] = data['Mv'] + modulo_distancia[k]
  Xiso = np.zeros_like(Yiso)
  Xiso[:] = data['(B-V)o'] + E
  ISO = np.dstack((Xiso,Yiso))
  ISO = np.unique(ISO,axis=1)
  A = np.zeros(len(ISO))
  B = np.zeros(len(ISO))
  for i in range(len(ISO)):
    for j in range(len(AGLO)):
      C,D = jpt(AGLO[j],ISO[i])
      final = frayn(ISO[i][C], ISO[i][D],AGLO[j])
      A[i] += final
      B[i] += final*10**(-0.4*AGLO[j][1])
  return A,B


#################


def regressao_aglomerado():
    x = XAglo
    y = YAglo
    regressao_inicial = linregress(x,y)
    coefs = [regressao_inicial.slope,regressao_inicial.intercept]
    coefs_erro = [regressao_inicial.stderr,regressao_inicial.intercept_stderr]
    t_fit = np.linspace(x.min(),x.max(),len(x))
    fit = linear_func(coefs,t_fit)
    sigma = np.sqrt((t_fit*coefs_erro[0])**2 + (coefs_erro[1])**2) #Intervalo Sigma
    xadj = []
    yadj = []
    count = 0
    ytentativa = coefs[0]*x + coefs[1]
    for element in y:
        if ytentativa[count] + 1*sigma[count] >= element and ytentativa[count] - 1*sigma[count] <= element:
            xadj.append(x[count])
            yadj.append(y[count])
        count+=1
    xadj = np.asarray(xadj)
    yadj = np.asarray(yadj)
    #estrelas_antes = len(x)
    #estrelas_depois = len(xadj)
    #print('Havia',estrelas_antes, 'estrelas antes do sigma-clipping.' )
    #print(estrelas_antes - estrelas_depois, 'estrelas foram retiradas.')
    #print('Apenas', estrelas_depois, 'remanesceram no intervalo 1-sigma.')
    #print(len(xadj)/len(x))
    regressao_mainseq = linregress(xadj,yadj)
    coefs_ms = [regressao_mainseq.slope,regressao_mainseq.intercept]
    coefs_erro_ms = [regressao_mainseq.stderr,regressao_mainseq.intercept_stderr]
    f = open("regressao_" + arquivo, "w")
    f.write("Slope,Intercept,Slope_Error,Intercept_Error,TurnOffColor\n")
    f.write( str(coefs_ms[0]) + ', ' + str(coefs_ms[1]) + ', ' + str(coefs_erro_ms[0]) + ', ' + str(coefs_erro_ms[1]) + ', ' + str(x.min()) + '\n')
    f.close()
def fit_inicial(show = False):
    global idade, distancia_estimada
    regressao_isocronas = pd.read_csv('../../../Isocronas/Regression_Iso.txt', header = 0)
    f1 = interp1d(regressao_isocronas['(B-V)TurnOff'],  regressao_isocronas['Age'],kind= 'linear')
    regressao_aglomerado = pd.read_csv('regressao_' + arquivo, header = 0)
    idade = f1(regressao_aglomerado['TurnOffColor'] - E)
    idade = np.around(idade,1)[0]
    isocro_idadeinicial = regressao_isocronas[regressao_isocronas['Age'] == idade]
    distancia_estimada = distancia(regressao_aglomerado['Slope'].item(), regressao_aglomerado['Intercept'].item(), isocro_idadeinicial['Intercept'].item(),E)
    if show == True:
        isocrona_idade_estimada = isocronas[isocronas['log(Age)']==idade]
        fig,ax = plt.subplots(figsize=(7,5)) #(figsize=(10,8))
        plt.gca().invert_yaxis()
        plt.plot(isocrona_idade_estimada['(B-V)o'] + E,isocrona_idade_estimada['Mv'] +5*np.log10(distancia_estimada/10)+3.1*E , label = 'Isócrona', color = 'r', zorder = 10)
        plt.scatter(aglomerado['B-V'],aglomerado['V'], label = nome, color = 'none', edgecolor = 'black')
        plt.legend(frameon=True)
        plt.xlabel(r"$(B-V)$")
        plt.ylabel(r"$V$")
        plt.title('Fit Inicial - ' + nome)
        plt.savefig('fit_inicial_' + nome + '.png', format = 'png')
        plt.tight_layout()
        plt.show();
def ajuste_inicial(show = False, show_final = False):
    modulodist_inicial = 5*np.log10(distancia_estimada/10) + E*3.1
    arrays_de_incremento = np.arange(0,3.05,0.05)
    subtracao_distancias = np.concatenate((-1*np.flip(arrays_de_incremento[1:]),arrays_de_incremento))
    global modulo_distancia
    modulo_distancia = subtracao_distancias + modulodist_inicial
    isocronas = pd.read_csv('../../../Isocronas/isocro.csv', header = 0)
    isocrona_idade_estimada = isocronas[isocronas['log(Age)']==idade]
    Yiso = np.zeros((len(modulo_distancia), len(isocrona_idade_estimada)))
    for i in range(len(modulo_distancia)):
      Yiso[i] = isocrona_idade_estimada['Mv'] + modulo_distancia[i]
    Xiso = np.zeros_like(Yiso)
    Xiso[:] = isocrona_idade_estimada['(B-V)o'] + E
    ISO = np.dstack((Xiso,Yiso))
    ISO = np.unique(ISO,axis=1)
    chisquared = np.zeros_like(modulo_distancia)
    Beauchamp = np.zeros_like(modulo_distancia)
    for i in range(len(ISO)):
      for j in range(len(AGLO)):
        C,D = jpt(AGLO[j],ISO[i])
        final = frayn(ISO[i][C], ISO[i][D],AGLO[j])
        chisquared[i] += final
        Beauchamp[i] += final*10**(-0.4*AGLO[j][1])
    min_beau = np.where(Beauchamp==min(Beauchamp))[0]
    min_chi = np.where(chisquared==min(chisquared))[0] ##Onde sao os minimos
    global minimo_beau, minimo_chi
    minimo_beau = modulo_distancia[min_beau]
    minimo_chi = modulo_distancia[min_chi]
    if show == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,6))
        ax1.scatter(modulo_distancia, Beauchamp, color = 'k')
        ax2.scatter(modulo_distancia, chisquared, color = 'k')
        ax1.set_xlabel(r'$ \mathbf{V - M_V}$')
        ax2.set_xlabel(r'$ \mathbf{V - M_V}$')
        ax1.set_ylabel('Função Beauchamp', fontweight = 'bold')
        ax2.set_ylabel(r'$ \mathbf{\chi^2}$')
        fig.suptitle(nome, fontweight = 'bold')
        plt.show();
    if show_final == True:
        fig,ax = plt.subplots(figsize=(10,8)) #(figsize=(10,8))
        plt.gca().invert_yaxis()
        plt.plot(isocrona_idade_estimada['(B-V)o'] + E, isocrona_idade_estimada['Mv'] + minimo_beau , label = 'Beauchamp', color = 'green', zorder = 10)
        plt.plot(isocrona_idade_estimada['(B-V)o'] + E, isocrona_idade_estimada['Mv'] + minimo_chi , label = r'$ \chi^2 $', color = 'red', zorder = 10)
        plt.plot(isocrona_idade_estimada['(B-V)o'] + E,isocrona_idade_estimada['Mv'] + modulodist_inicial , '--', label = 'Isócrona Inicial', color = 'magenta', zorder = 10)
        plt.plot(isocrona_idade_estimada['(B-V)o'] + E,isocrona_idade_estimada['Mv'] + modulo_teorico , '--', label = 'Teórica', color = 'blue', zorder = 10)
        plt.scatter(aglomerado['B-V'] ,aglomerado['V'], label = nome, color = 'none', edgecolor = 'black')
        plt.legend(frameon=True)
        plt.xlabel(r"$ \mathbf{(B-V)}$")
        plt.ylabel(r"$ \mathbf{V}$");
        plt.show();
def n_idades(show = False):
    BeauchampAGES = np.zeros_like(idades)
    chisquaredAGES = np.zeros_like(idades)
    for j in range(len(idades)):
        data = isocronas[isocronas['log(Age)'] == idades[j]]
        Xiso = np.array(data['(B-V)o'] + E)
        Yiso = np.array(data['Mv'] + minimo_chi)
        ISO = (np.vstack((Xiso,Yiso)).T)
        ISO = np.unique(ISO, axis=0)
        for i in range(len(AGLO)):
            C,D = jpt(AGLO[i],ISO)
            final = frayn(ISO[C], ISO[D],AGLO[i])
            chisquaredAGES[j] += final
            BeauchampAGES[j] += final*10**(-0.4*AGLO[i][1])
    if show == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,6))
        ax1.scatter(idades, BeauchampAGES, color = 'none', edgecolor = 'red', s = 20)
        ax2.scatter(idades, chisquaredAGES, color = 'none', edgecolor = 'blue', s = 20)
        ax1.set_xlabel('log(Age)',  fontweight = 'bold')
        ax2.set_xlabel('log(Age)',  fontweight = 'bold')
        ax1.set_ylabel('Função Beauchamp',  fontweight = 'bold')
        ax2.set_ylabel(r'$ \mathbf{\chi^2}$')
        fig.suptitle(nome, fontweight = 'bold')
        plt.show();
def final(show = False):
    newage = idades#[5:25]
    resultado_chi = []
    resultado_beau = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
      results = executor.map(chi_to_age,newage)
    for i in results:
        resultado_chi.append(i[0])
        resultado_beau.append(i[1])
    resultado_chi = np.array(resultado_chi)
    resultado_beau = np.array(resultado_beau)
    locais_chi = np.where(resultado_chi==np.min(resultado_chi))
    locais_beau= np.where(resultado_beau==np.min(resultado_beau))
    print('Melhor resultado pelo método de Chi: ')
    print('log(Age) = ', newage[locais_chi[0][0]])
    print('V - M_V = ', modulo_distancia[locais_chi[1][0]])
    print('\n')
    print('Melhor resultado pelo método de Beauchamp: ')
    print('log(Age) = ', newage[locais_beau[0][0]])
    print('V - M_V = ', modulo_distancia[locais_beau[1][0]])
    print('\n')
    print('log(Age) Teórica: ', idade_teorica)
    print('V - M_V Observado: ', modulo_teorico)
    if show == True:
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import ImageGrid
        x = modulo_distancia
        y = newage
        cmap = cm.get_cmap('jet')
        cmap = cm.jet
        fig, ax = plt.subplots(figsize = (8,6)) #(figsize=(10,8))
        levels = 200
        im  = ax.contourf(x, y, resultado_chi, levels= levels, antialiased=False, cmap=cmap)
        cbar = fig.colorbar(im)
        cbar.set_label(r'$ \mathbf{\chi^2}$', fontweight = 'bold', rotation=0, labelpad=15)
        ax.set_xlabel(r'$ \mathbf{V - M_V}$', fontweight = 'bold', labelpad=10)
        ax.set_ylabel('log(Age)', fontweight = 'bold', labelpad=10)
        ax.set_title(nome, fontweight = 'bold')
        plt.savefig('chi_final' + nome.strip() +'.png', format = 'png')
        plt.show();

        fig, ax = plt.subplots(figsize = (8,6)) #(figsize=(10,8))
        levels = 200
        im  = ax.contourf(x, y, resultado_beau, levels= levels, antialiased=False, cmap=cmap)
        cbar = fig.colorbar(im)
        cbar.set_label('Beauchamp', fontweight = 'bold', rotation=270, labelpad=15)
        ax.set_xlabel(r'$ \mathbf{V - M_V}$', fontweight = 'bold', labelpad=10)
        ax.set_ylabel('log(Age)', fontweight = 'bold', labelpad=10)
        ax.set_title(nome, fontweight = 'bold')
        plt.savefig('beauchamp_final_' + nome.strip() +'.png', format = 'png')
        plt.show();

T = True
F = False


regressao_aglomerado()
fit_inicial(show = T)
ajuste_inicial(show = T, show_final = T)
n_idades(show = T)
final(show = T)
