import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
from multiprocessing import get_start_method
from multiprocessing import get_context
import time 

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
    return h**2 
def dist(X,Y):
  distancia = 0
  for j in range(len(X)):
    distancia+= (X[j] - Y[j])**2
  return distancia


aglomerado = pd.read_csv('membros/final_semradec.csv')
XAglo = aglomerado['bp_rp']
YAglo = aglomerado['phot_g_mean_mag']
AGLO = np.vstack((XAglo,YAglo)).T

isocronas_geral = pd.read_csv('../../../Isocronas_Gaia/gaia_metal.csv')
isocronas_geral = isocronas_geral[(isocronas_geral.label <= 4) & (isocronas_geral.MH == -0.2)]
isocronas_geral['BP_RP'] = isocronas_geral['G_BPmag'] - isocronas_geral['G_RPmag']
idades = np.unique(isocronas_geral['logAge'])
newage = idades[4:10]
#metalicidades = np.unique(isocronas_geral.MH)

avs = np.arange(0.15,0.3,0.01)
passo = 0.01
modulodist_inicial = 13
modulo_distancia = np.arange(modulodist_inicial - 1, modulodist_inicial + 1, passo)

arrays = np.array(np.meshgrid(newage,avs, modulo_distancia)).T.reshape(-1, 3)


def fit_all(age, Av, distance_modulus): 
    data = isocronas_geral[(isocronas_geral.logAge == age) ]
    Yiso = np.zeros((1, len(data)))
    for k in range(1):
        Yiso[k] = data['Gmag'] + distance_modulus + 0.83627*Av
    Xiso = np.zeros_like(Yiso)
    Xiso[:] = data['BP_RP'] + Av*(1.08337 - 0.63439)
    ISO = np.dstack((Xiso,Yiso))
    ISO = np.unique(ISO,axis=1)
    A = np.zeros(len(ISO))
    for i in range(len(ISO)):
        for j in range(len(AGLO)):
            C,D = jpt(AGLO[j],ISO[i])
            final = frayn(ISO[i][C], ISO[i][D],AGLO[j])
            A[i] += final
    return A
def teste_atomico(array):
    if __name__ == '__main__':
        # get the start method
        method = get_start_method()
        print(f'Main process using {method}')
        # create a process context
        ctx = get_context('fork')
        # create and configure the process pool
        with Pool(context=ctx) as pool:
            results = pool.starmap(fit_all,array)
    resultado_final = []
    for i in results:
        resultado_final.append(i)
    return resultado_final

start = time.time()

chis = teste_atomico(arrays)
chis = np.array(chis)
end = time.time()

print('Tempo = ', end-start)

np.savetxt('output_boss/chis_quadrados_teste.txt', chis)

local = np.where(chis == np.min(chis))[0][0]

print(arrays[local], np.min(chis))

def write_file():
    f = open('output_boss/testechi_parametros.csv', 'w') 
    f.write('Chi,Age,Av,ModDist\n')
    for i in range(len(chis)):
        chi = chis[i][0]
        idade = arrays[i][0]
        av = arrays[i][1]
        dist = arrays[i][2]
        f.write(str(chi) + ',' + str(idade) + ',' + str(av) + ',' + str(dist) +'\n')
    f.close()

write_file()



