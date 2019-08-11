"""
Josué Alexis M.G.
09-08-19
Decision Tree Classifier
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"se carga la base de datos en una variable"
"it´s loaded the database in a variable"
df = pd.read_csv('C:\\atletasO.csv')
print(df.head(3))
df.loc[df.Medal.isnull(), 'Medal']= "NO"
df = df.dropna() 
"elemina valores no completados totalmente" 
"deletes uncompleted values completely"

"árbol con profundidad de 3"
"tree with depth of 3"
arbol = DecisionTreeClassifier (criterion='entropy', max_depth=3)

"variables para entrenamiento"
"variables to training"
xtrain = df[['Age','Height','Weight']]
ytrain = df['Medal']
arbol.fit(xtrain,ytrain)

print(arbol.classes_)

"""export_graphviz(arbol, out_file='medallas.dot')"""
"para exportar desde la terminar la forma de tu árbol"
"to export from the terminal the shape of your tree"

arbol2 = DecisionTreeClassifier (criterion='entropy', max_depth=3, class_weight = 'balanced')
arbol2.fit(xtrain,ytrain)
print(arbol2.classes_)
"Segundo árbol, pero con valores balanceados"
"second tree with balanced values"

"""export_graphviz(arbol2, out_file='medallas2.dot')"""

columnasUsadas = df[['Age','Height','Weight','Medal']]

datos_prueba = columnasUsadas.sample(n=10, random_state=1)

predicciones = arbol.predict(datos_prueba[['Age','Height','Weight']])
predicciones2 = arbol2.predict(datos_prueba[['Age','Height','Weight']])

"presición de mi sistema"
"acuracy of my system"
accuracy_score(datos_prueba['Medal'],predicciones)
accuracy_score(datos_prueba['Medal'],predicciones2)

confusion_matrix(datos_prueba['Medal'], predicciones)
confusion_matrix(datos_prueba['Medal'], predicciones2)