import dagshub
import sklearn
import mlflow 
import mlflow.sklearn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sklearn.utils
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

iris:sklearn.utils.Bunch = load_iris()


# dags hub intialization
dagshub.init(
    repo_owner='kameshkotwani',
    repo_name='mlflow-dsmp',
    mlflow=True
)


X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# params for RandomForest Model
MAX_DEPTH = 5
N_ESTIMATORS = 3

# mlflow.set_experiment("decision-tree-iris") # this makes the folder if it does not exist
mlflow.set_tracking_uri("https://dagshub.com/kameshkotwani/mlflow-dsmp.mlflow")

with mlflow.start_run() as run:

    dt = DecisionTreeClassifier(max_depth= MAX_DEPTH)
    
    dt.fit(X_train,y_train)
    
    y_pred = dt.predict(X_test)
    
    accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)


    mlflow.log_metric("accuracy",accuracy)

    mlflow.log_param("max_depth",MAX_DEPTH)

    print(accuracy)

    print('accuracy', accuracy)
    
# Log confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    #save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    
    #log the model
    mlflow.log_artifact(__file__)
    
    
    mlflow.sklearn.log_model(dt, "decision_tree_model")


    mlflow.set_tag('author', 'kamesh')
    mlflow.set_tag('project', 'iris-classification')
    mlflow.set_tag('algorithm', 'decision-tree')


