import sklearn
import mlflow 
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sklearn.utils
from sklearn.ensemble import RandomForestClassifier
iris:sklearn.utils.Bunch = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# params for RandomForest Model
MAX_DEPTH = 15
N_ESTIMATORS = 10

with mlflow.start_run() as run:

    rf = RandomForestClassifier(max_depth= MAX_DEPTH,n_estimators = N_ESTIMATORS)
    
    rf.fit(X_train,y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)


    mlflow.log_metric("accuracy",accuracy)

    mlflow.log_param("max_depth",MAX_DEPTH)
    mlflow.log_param("n_estimators",N_ESTIMATORS)

    print(accuracy)



