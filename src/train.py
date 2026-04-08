from sklearn.datasets import load_iris
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/churn.csv')
x , y = df.drop('Churn', axis=1), df['Churn']
x,xt,y,yt = train_test_split(x,y , test_size=0.2)

e = 1
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(x,y)
    acc = model.score(xt,yt)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(model,artifact_path='model', registered_model_name='churn')
