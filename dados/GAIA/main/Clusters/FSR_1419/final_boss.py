import numpy as np
import pandas as pd

from multiprocessing.pool import Pool
from multiprocessing import get_start_method
from multiprocessing import get_context

def jpt(X,Y):
    array = (X - Y)
    array = np.sum(array**2, axis = 1)
    A, B = np.argpartition(array, 1)[0:2]
    return A,B
def frayn(X1,X2,X3):
    if X1[0]==X2[0] and X1[1] == X2[1]:
        h = 0
        print('Warning! Equal values on isochrones')
    else:
        h = 2*((X3[0]-X1[0])*(X2[1]-X1[1]) - (X3[1]-X1[1])*(X2[0]-X1[0]))/np.sqrt(dist(X1,X2))
    return h 
def dist(X,Y):
  distancia = 0
  for j in range(len(X)):
    distancia+= (X[j] - Y[j])**2
  return distancia


aglomerado = pd.read_csv('membros/raio_phasespace.csv')
XAglo = aglomerado['bp_rp']
YAglo = aglomerado['phot_g_mean_mag']
AGLO = np.vstack((XAglo,YAglo)).T

isocronas_geral = pd.read_csv('../../../Isocronas_Gaia/gaia_metal.csv')
isocronas_geral = isocronas_geral[isocronas_geral.label <= 4]
isocronas_geral['BP_RP'] = isocronas_geral['G_BPmag'] - isocronas_geral['G_RPmag']
idades = np.unique(isocronas_geral['logAge'])
newage = idades[4:11]
metalicidades = np.unique(isocronas_geral.Zini)


avs = np.arange(1,2.25,0.05)
passo = 0.05
modulodist_inicial = 15
arrays_de_incremento = np.arange(0,1.55,passo)
subtracao_distancias = np.concatenate((-1*np.flip(arrays_de_incremento[1:]),arrays_de_incremento))
modulo_distancia = subtracao_distancias + modulodist_inicial


def loucura(IDADE):
    endval = []
    for z in range(len(metalicidades)):
        isocronas = isocronas_geral[isocronas_geral.Zini == metalicidades[z]]
        data = isocronas[isocronas['logAge'] == IDADE]
        Yiso = np.zeros((len(modulo_distancia), len(avs), len(data)))
        for k in range(len(modulo_distancia)):
            for j in range(len(avs)):
                Yiso[k][j][:] = modulo_distancia[k] + 0.83627*avs[j] + data['Gmag']
        Xiso = np.zeros_like(Yiso)
        for k in range(len(modulo_distancia)):
            for j in range(len(avs)):
                Xiso[k][j][:] = data['BP_RP'] + avs[j]*(1.08337 - 0.63439)
        ISO = np.stack((Xiso, Yiso), axis = 3)
        ISO = np.unique(ISO, axis = 2)
        A = np.zeros((len(ISO),len(avs)))
        for i in range(len(ISO)):
            for w in range((len(avs))):
                isocrona = ISO[i][w]
                for j in range(len(AGLO)):
                    C,D = jpt(AGLO[j], isocrona)
                    final = frayn(isocrona[C], isocrona[D], AGLO[j])
                    final = final if np.isnan(final) == False else 0
                    A[i][w] +=  final
        endval.append(A.T)
    return endval

def teste_atomico():
    if __name__ == '__main__':
        # get the start method
        method = get_start_method()
        print(f'Main process using {method}')
        # create a process context
        ctx = get_context('fork')
        # create and configure the process pool
        with Pool(context=ctx) as pool:
            results = pool.map(loucura,newage)
    global resultado_final
    resultado_final = []
    for i in results:
        resultado_final.append(i)


teste_atomico()

locais = (np.where(resultado_final == np.min(resultado_final)))

idade = newage[locais[0][0]]
zini = metalicidades[locais[1][0]]
av = avs[locais[2][0]]
mod = modulo_distancia[locais[3][0]]

print('log(Age) = ', idade)
print('Z = ', zini)
print('Av = ', av)
print('Modulo de distancia = ', mod)

file = open('output_boss/JPT_MONSTRO.txt',"w")
file.write(str(resultado_final))
file.close()