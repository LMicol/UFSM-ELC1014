" Imports para utilização de MiniSom "
from minisom import MiniSom
from sklearn.datasets import load_breast_cancer
import time


" Parâmetros "
ksom_linhas  = 20
ksom_colunas = 30
iterations   = 500
sigma = 1
aprendizado = 0.5


" Dados "
data, target = load_breast_cancer(True)


" Criação do Mapa "
KSOM = MiniSom( x = ksom_linhas, y = ksom_colunas,
                input_len=data.shape[1], sigma=sigma,
                learning_rate=aprendizado )
KSOM.random_weights_init(data)


" Treinamento "
inicio = time.time()
KSOM.train_random(data, iterations)
fim = time.time()
print("Tempo total: ", (fim-inicio), " segundos" )


" Plot "
'from pylab import plot, axis, show, pcolor, colorbar, bone'
from pylab import *
bone()
pcolor(KSOM.distance_map().T)
colorbar()
  
markers = ['v','^','*']
colors  = ['r','g','b']
for cnt,xx in enumerate(data):
    w = KSOM.winner(xx)
    plot( w[0]+.5, w[1]+.5, markers[target[cnt]] , markerfacecolor = 'none',
          markeredgecolor = colors[target[cnt]], markersize=7, markeredgewidth = 2)
    
axis([0,KSOM._weights.shape[0],0,KSOM._weights.shape[1]])
show()


