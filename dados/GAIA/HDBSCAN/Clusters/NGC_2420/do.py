import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
import concurrent.futures
import multiprocessing as mp
from scipy.stats import linregress
from scipy.interpolate import interp1d
#import matplotlib.font_manager as fm
from matplotlib.ticker import AutoMinorLocator


#font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams['axes.linewidth'] = 1.5


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 13}

plt.rc('font', **font)

nome = 'NGC 2420'
def rename(string):
    return string.replace(' ', '')


modulo_teorico = 5*np.log10(2435) - 5 ###
idade_teorica = 9.3

arquivo = 'final_5d.csv'

def global_var(x):
    global aglomerado, isocronas, E, idades, XAglo, YAglo, AGLO, Av
    AVNN = 0.117
    E = (1.09909-0.63831)*AVNN
    Av = 0.83139*AVNN
    aglomerado =  pd.read_csv(x)
    aglomerado = aglomerado.dropna(how = 'any', subset=['bp_rp','phot_g_mean_mag'])
    aglomerado = aglomerado.reset_index()
    #isocronas = pd.read_csv('../../../Isocronas/isocro.csv', header = 0)
    isocronas = pd.read_csv('../iso_gaia_clipped.csv')
    idades = np.unique(isocronas['logAge'])
    XAglo = aglomerado['bp_rp']
    YAglo = aglomerado['phot_g_mean_mag']
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
  data = isocronas[isocronas['logAge'] == IDADE]
  Yiso = np.zeros((len(modulo_distancia), len(data)))
  for k in range(len(modulo_distancia)):
    Yiso[k] = data['Gmag'] + modulo_distancia[k]
  Xiso = np.zeros_like(Yiso)
  Xiso[:] = data['BP-RP'] + E
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
print('log(Age) Teórica: ', idade_teorica)
print('V - M_V Observado: ', modulo_teorico)
print('\n')

def regressao_aglomerado(n_sigma = 2):
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
        if ytentativa[count] + n_sigma*sigma[count] >= element and ytentativa[count] - n_sigma*sigma[count] <= element:
            xadj.append(x[count])
            yadj.append(y[count])
        count+=1
    xadj = np.asarray(xadj)
    yadj = np.asarray(yadj)
    regressao_mainseq = linregress(xadj,yadj)
    coefs_ms = [regressao_mainseq.slope,regressao_mainseq.intercept]
    coefs_erro_ms = [regressao_mainseq.stderr,regressao_mainseq.intercept_stderr]
    cor_to = np.min(xadj)
    mag_to = yadj[np.where(xadj==cor_to)[0][0]]
    f = open("regressao_" + arquivo, "w")
    f.write("Slope,Intercept,Slope_Error,Intercept_Error,TurnOffColor,TurnOffMag\n")
    f.write( str(coefs_ms[0]) + ', ' + str(coefs_ms[1]) + ', ' + str(coefs_erro_ms[0]) + ', ' + str(coefs_erro_ms[1]) + ', ' + str(cor_to) + ', ' + str(mag_to) + '\n')
    f.close()
def fit_inicial(show = False):
    global idade_inicial, distancia_estimada
    regressao_isocronas = pd.read_csv('../Regressoes_Isocronas_Gaia.txt', header = 0)
    f1 = interp1d(regressao_isocronas['(BP-RP)TurnOff'],  regressao_isocronas['Age'],kind= 'linear')
    regressao_aglomerado = pd.read_csv('regressao_' + arquivo, header = 0)
    idade_inicial = f1(regressao_aglomerado['TurnOffColor'] - E)
    idade_inicial = np.around(idade_inicial,1)[0]
    print('Idade inicial estimada:')
    print(idade_inicial)
    isocro_idadeinicial = regressao_isocronas[regressao_isocronas['Age'] == idade_inicial] ##mudando aqui
    global slope_regressao
    slope_regressao = regressao_aglomerado['Slope'].item()
    distancia_estimada = distancia(regressao_aglomerado['Slope'].item(), regressao_aglomerado['Intercept'].item(), isocro_idadeinicial['Intercept'].item(),E)
    print('Modulo de distancia inicial estimado:')
    print( 5*np.log10(distancia_estimada/10))
    print('\n')
    if show == True:
        isocrona_idade_estimada = isocronas[isocronas['logAge']==idade_inicial]
        fig,ax = plt.subplots(figsize=(7,5)) #(figsize=(10,8))
        plt.gca().invert_yaxis()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)
        ax.plot(isocrona_idade_estimada['BP-RP'] + E,isocrona_idade_estimada['Gmag'] +5*np.log10(distancia_estimada/10)+Av , label =  'log(Age) = ' + str(idade_inicial), color = 'r', zorder = 10)
        ax.scatter(XAglo,YAglo, color = 'none', edgecolor = 'black')
        ax.set_xlabel(r"$ \mathbf{BP-RP}$")
        ax.set_ylabel(r"$\mathbf{G}$")
        ax.legend()
        fig.suptitle('Fit Inicial - ' + nome , fontweight = 'bold')
        plt.savefig('fit_inicial_' + rename(nome) + '.png', format = 'png')
        plt.tight_layout()
        plt.show();
def ajuste_inicial(show = False, show_final = False):
    modulodist_inicial = 5*np.log10(distancia_estimada/10) + E*3.1
    arrays_de_incremento = np.arange(0,3.05,0.05)
    subtracao_distancias = np.concatenate((-1*np.flip(arrays_de_incremento[1:]),arrays_de_incremento))
    global modulo_distancia
    modulo_distancia = subtracao_distancias + modulodist_inicial
    isocrona_idade_estimada = isocronas[isocronas['logAge']==idade_inicial]
    Yiso = np.zeros((len(modulo_distancia), len(isocrona_idade_estimada)))
    for i in range(len(modulo_distancia)):
      Yiso[i] = isocrona_idade_estimada['Gmag'] + modulo_distancia[i]
    Xiso = np.zeros_like(Yiso)
    Xiso[:] = isocrona_idade_estimada['BP-RP'] + E
    ISO = np.dstack((Xiso,Yiso))
    ISO = np.unique(ISO,axis=1)
    chisquared = np.zeros_like(modulo_distancia)
    Beauchamp = np.zeros_like(modulo_distancia)
    for i in range(len(ISO)):
      for j in range(len(AGLO)):
        C,D = jpt(AGLO[j],ISO[i])
        final = frayn(ISO[i][C], ISO[i][D],AGLO[j])
        chisquared[i] += final
        Beauchamp[i] += final*10**(-0.6*AGLO[j][1])
    min_beau = np.where(Beauchamp==min(Beauchamp))[0]
    min_chi = np.where(chisquared==min(chisquared))[0] ##Onde sao os minimos
    global minimo_beau, minimo_chi
    minimo_beau = modulo_distancia[min_beau]
    minimo_chi = modulo_distancia[min_chi]
    if show == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,6))
        ax1.scatter(modulo_distancia, Beauchamp, color = 'k')
        ax2.scatter(modulo_distancia, chisquared, color = 'k')

        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax1.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax1.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax1.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)

        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax2.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax2.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax2.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)

        ax1.set_xlabel(r'm - M')
        ax2.set_xlabel(r'm - M')
        ax1.set_ylabel('Função Beauchamp', fontweight = 'bold')
        ax2.set_ylabel(r'$ \mathbf{\chi^2}$')
        fig.suptitle(nome, fontweight = 'bold')
        plt.show();
    if show_final == True:
        fig,ax = plt.subplots(figsize=(10,8)) #(figsize=(10,8))
        plt.gca().invert_yaxis()
        isocrona_teorica= isocronas[isocronas['logAge']==idade_teorica]
        ax.plot(isocrona_idade_estimada['BP-RP'] + E, isocrona_idade_estimada['Gmag'] + minimo_beau , label = 'Beauchamp', color = 'green', zorder = 10)
        ax.plot(isocrona_idade_estimada['BP-RP'] + E, isocrona_idade_estimada['Gmag'] + minimo_chi , label = r'$ \chi^2 $', color = 'red', zorder = 10)
        ax.plot(isocrona_idade_estimada['BP-RP'] + E,isocrona_idade_estimada['Gmag'] + modulodist_inicial , '--', label = 'Isócrona Inicial', color = 'blue', zorder = 10)
        ax.scatter(XAglo,YAglo, color = 'none', edgecolor = 'black')

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)

        ax.legend(frameon=True)
        fig.suptitle( 'Ajuste Inicial - '+ nome + '\n log(Age) = ' + str(idade_inicial), fontweight = 'bold')
        ax.set_xlabel(r"$ \mathbf{BP-RP}$")
        ax.set_ylabel(r"$ \mathbf{G}$")
        plt.savefig('ajuste_inicial_'+ rename(nome) + '.png', format = 'png')
        plt.tight_layout()
        plt.show();
def n_idades(show = False):
    BeauchampAGES = np.zeros_like(idades)
    chisquaredAGES = np.zeros_like(idades)
    for j in range(len(idades)):
        data = isocronas[isocronas['logAge'] == idades[j]]
        Xiso = np.array(data['BP-RP'] + E)
        Yiso = np.array(data['Gmag'] + minimo_chi)
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

        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax1.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax1.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax1.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)

        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
        ax2.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
        ax2.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
        ax2.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)

        ax1.set_xlabel('log(Age)',  fontweight = 'bold')
        ax2.set_xlabel('log(Age)',  fontweight = 'bold')
        ax1.set_ylabel('Função Beauchamp',  fontweight = 'bold')
        ax2.set_ylabel(r'$ \mathbf{\chi^2}$')
        fig.suptitle(nome + '\n m - M fixado = ' + str(minimo_chi) + ' (minimo chi)', fontweight = 'bold')
        plt.tight_layout()
        plt.show();
def final(show = False):
    newage = idades
    resultado_chi = []
    resultado_beau = []
    with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
      results = executor.map(chi_to_age,newage)
    for i in results:
        resultado_chi.append(i[0])
        resultado_beau.append(i[1])
    resultado_chi = np.array(resultado_chi)
    resultado_beau = np.array(resultado_beau)
    locais_chi = np.where(resultado_chi==np.min(resultado_chi))
    locais_beau= np.where(resultado_beau==np.min(resultado_beau))
    global idadechi,idadebeau,distchi,distbeau
    idadechi = newage[locais_chi[0][0]]
    distchi = modulo_distancia[locais_chi[1][0]]
    idadebeau = newage[locais_beau[0][0]]
    distbeau = modulo_distancia[locais_beau[1][0]]
    print('Melhor resultado pelo método de Chi: ')
    print('log(Age) = ', idadechi)
    print('V - M_V = ', distchi)
    print('\n')
    print('Melhor resultado pelo método de Beauchamp: ')
    print('log(Age) = ', idadebeau)
    print('V - M_V = ', distbeau)
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
        ax.set_xlabel('m - M', fontweight = 'bold', labelpad=10)
        ax.set_ylabel('log(Age)', fontweight = 'bold', labelpad=10)
        ax.set_title(nome, fontweight = 'bold')
        plt.savefig('chi_final_' + rename(nome) +'.png', format = 'png')
        plt.show();

        fig, ax = plt.subplots(figsize = (8,6)) #(figsize=(10,8))
        levels = 200
        im  = ax.contourf(x, y, resultado_beau, levels= levels, antialiased=False, cmap=cmap)
        cbar = fig.colorbar(im)
        cbar.set_label('Beauchamp', fontweight = 'bold', rotation=270, labelpad=15)
        ax.set_xlabel('m - M', fontweight = 'bold', labelpad=10)
        ax.set_ylabel('log(Age)', fontweight = 'bold', labelpad=10)
        ax.set_title(nome, fontweight = 'bold')
        plt.savefig('beauchamp_final_' + rename(nome) +'.png', format = 'png')
        plt.show();
def plot_finalchi():
    isocrona_chi = isocronas[isocronas['logAge']==idadechi]
    fig,ax = plt.subplots(figsize=(7,5))
    plt.gca().invert_yaxis()
    ax.plot(isocrona_chi['BP-RP'] + E, isocrona_chi['Gmag'] + distchi +Av, label = 'log(Age) = ' + str(idadechi), color = 'r', zorder = 10)
    ax.scatter(XAglo,YAglo, color = 'none', edgecolor = 'black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
    ax.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)
    ax.legend(frameon=True)
    ax.set_xlabel(r"$ \mathbf{BP-RP}$")
    ax.set_ylabel(r"$\mathbf{G}$")
    fig.suptitle('Fit Final Chi - ' + nome, fontweight = 'bold')
    plt.savefig('fit_final_chi' + rename(nome) + '.png', format = 'png')
    plt.tight_layout()
    plt.show();
def plot_finalbeau():
    isocrona_chi = isocronas[isocronas['logAge']==idadebeau]
    fig,ax = plt.subplots(figsize=(7,5))
    plt.gca().invert_yaxis()
    ax.plot(isocrona_chi['BP-RP'] + E, isocrona_chi['Gmag'] + distbeau +Av, label = 'log(Age) = ' + str(idadebeau), color = 'r', zorder = 10)
    ax.scatter(XAglo,YAglo, color = 'none', edgecolor = 'black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
    ax.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)
    ax.legend(frameon=True)
    ax.set_xlabel(r"$ \mathbf{BP-RP}$")
    ax.set_ylabel(r"$\mathbf{G}$")
    fig.suptitle('Fit Final Beauchamp - ' + nome, fontweight = 'bold')
    plt.savefig('fit_final_beau' + rename(nome) + '.png', format = 'png')
    plt.tight_layout()
    plt.show();
def plot_teorico():
    isocrona_chi = isocronas[isocronas['logAge']==idade_teorica]
    fig,ax = plt.subplots(figsize=(7,5))
    plt.gca().invert_yaxis()
    ax.plot(isocrona_chi['BP-RP'] + E, isocrona_chi['Gmag'] + modulo_teorico + Av , label = 'log(Age) = ' + str(idade_teorica), color = 'r', zorder = 10)
    ax.scatter(XAglo,YAglo, color = 'none', edgecolor = 'black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which = 'major', axis = 'y', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'y', direction='in', length = 4)
    ax.tick_params(which = 'major', axis = 'x', direction='in', length = 7)
    ax.tick_params(which = 'minor', axis = 'x', direction='in', length = 4)
    ax.legend()
    ax.set_xlabel(r"$ \mathbf{BP-RP}$")
    ax.set_ylabel(r"$\mathbf{G}$")
    fig.suptitle('Fit Teórico - ' + nome, fontweight = 'bold')
    plt.savefig('fit_teorico' + rename(nome) + '.png', format = 'png')
    plt.tight_layout()
    plt.show();


T = True
F = False

#plot_teorico()
regressao_aglomerado()
fit_inicial(show = F)
ajuste_inicial(show = F, show_final = F)
n_idades(show = F)
final(show = F)


#plot_finalchi()
#plot_finalbeau()

file = open('parametros_finais.csv',"w")
file.write('#Aglomerado ' + nome + ', Av = ' + str(Av) + '\n')
file.write('Metodo,Age,ModDist\n')
file.write('DIAS2022, ' + str(idade_teorica) + ', ' + str(modulo_teorico) +  '\n')
file.write('Inicial, ' + str(idade_inicial) + ', ' + str(5*np.log10(distancia_estimada/10)) +  '\n')
file.write('Chi, ' + str(idadechi) + ', ' + str(distchi) +  '\n')
file.write('Beauchamp, ' + str(idadebeau) + ', ' + str(distbeau))
file.close()