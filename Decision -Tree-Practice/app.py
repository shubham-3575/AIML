from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn import metrics #for evaluation

diabetes = pd.read_csv("diabetes.csv")
print(diabetes.columns)

X = diabetes.drop(['SkinThickness','Outcome'],axis=1) #drop the skin thisckness and outcome as they are not used for any classificati0on or regression
y = diabetes['Outcome']

#Splitting the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=20)

#Create Decision tree classifier Object
clf = DecisionTreeClassifier()

#Train decision tree classifier
clf.fit(X_train,y_train)

ypr = clf.predict(X_test)

metrics.accuracy_score(y_test,ypr)*100

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf,out_file = dot_data,
            	filled = True, rounded = True,
                special_characters = True, 
                feature_names =X.columns, class_names = ['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
