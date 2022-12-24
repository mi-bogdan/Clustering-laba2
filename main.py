import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler

def createDiagrams(data,optics_model,labels):
    # создаем массив и сохраняет туда последовательность элементов, где len(x) = кол-во элементов(строк) в матрице
    space = np.arange(len (data))
    # Сохранение расстояния достижимости до каждой точки
    reachability = optics_model.reachability_[optics_model.ordering_]
    
    # создаем фигуру с заданными размерами
    plt.figure(figsize = ( 20 , 9 ))
    # создаем Макет сетки для размещения подграфиков в фигуре; задаем Количество строк и столбцов сетки 
    #(представляет собой будто двумерный массив, в одну из ячеек которого можно разместить подграфик)
    G = gridspec.GridSpec( 1 , 2 )
    #Добавляем оси к тек. фигуре 
    ax2 = plt.subplot(G[ 0 , 1])
    ax1 = plt.subplot(G[ 0 , 0 ])
     
    colors = ['#FF0000', '#C71585', '#FF7F50', '#FF4500', '#FFA500', '#FFFF00', '#FF00FF', '#9370DB', '#8A2BE2', '#8B008B', '#4B0082'
         , '#483D8B', '#BC8F8F', '#2F4F4F', '#800000', '#0000FF', '#000080', '#00CED1']

    #ПОСТРОЕНИЕ1 графика достижимости-расстояния 
    for Class, colour in zip ( range ( 0 , len(np.unique(labels))), colors):
        Xk = space[labels == Class] # возращает id объекта текущего кластера, которому он принадлежит
        Rk = reachability[labels == Class]
        ax1.plot(Xk, Rk, colour, alpha = 0.8 )
    
    ax1.plot(space[labels == - 1 ], reachability[labels == - 1 ], 'k.' , alpha = 0.5 )
    ax1.set_ylabel( 'Reachability Distance' )
    ax1.set_title( 'Reachability Plot' )

    # ПОСТРОЕНИЕ2  OPTICS Clustering
    for Class, c in zip ( range ( 0 , len(np.unique(labels))), colors):
        Xk = data[optics_model.labels_ == Class]
        ax2.plot(Xk.iloc[:, 1 ], Xk.iloc[:, 2 ], 'o', color = c,  alpha = 0.7)

    
    ax2.plot(data.iloc[optics_model.labels_ == - 1 , 1 ], data.iloc[optics_model.labels_ == - 1 , 2 ],'k+' , alpha = 0.6 )

    ax2.set_title( 'OPTICS Clustering' ) 



dataTraining = pd.read_csv('hayes-roth.csv', nrows=90)
dataTest =  pd.read_csv('hayes-roth.csv', header=None, skiprows = 91)

#удаляем столбец Class, данные по которому не нужны нам для кластеризации
dataTraining.drop (dataTraining.columns[5], axis= 1 , inplace= True )
print('Обучающие данные')   
print(dataTraining)

#удаляем столбец Class, данные по которому не нужны нам для кластеризации
dataTest.drop(columns = 5,axis = 1, inplace=True)
print('Тренировочные данные')        
print(dataTest)

#определяем модель 
optics_model = OPTICS()
# тренируем модель
optics_model.fit(dataTraining)
labels = optics_model.labels_[optics_model.ordering_]
optics_model.ordering_ # сгруппированные id объектов, которые относятся к своим кластерам

# сохранение меток кластеров для каждого объекта (к которым  относятся) ; -1 означает шум 
createDiagrams(dataTraining,optics_model,labels)
print (labels)

labels = optics_model.fit_predict(dataTest)
labels = labels[optics_model.ordering_]
createDiagrams(dataTest,optics_model,labels)

print(" \nТаблица экспериментов подбора гиперпараметров \n ")

i=0
for minsamples, eps in zip ( range(2,14,2), np.arange(0.1,0.7,0.1)):
    optics_model = OPTICS(min_samples=minsamples, eps=eps)
    optics_model.fit(dataTest)
    labels = optics_model.fit_predict(dataTest)
    labels = labels[optics_model.ordering_]
    print('i=',i,' ', 'min_samples=',minsamples, '; eps=',eps,'\n')
    createDiagrams(dataTest,optics_model,labels)
    i+=1
