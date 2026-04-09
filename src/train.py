from sklearn.datasets import load_iris
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()

df = pd.read_csv('data/churn.csv')
x , y = df.drop('Churn', axis=1), df['Churn']
x,xt,y,yt = train_test_split(x,y , test_size=0.2)


x.to_csv("data/train_data.csv", index=False)

e = 1
with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(x,y)
    acc = model.score(xt,yt)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(model,artifact_path='model', registered_model_name='churn')
    run_id = run.info.run_id




# Get latest version 
latest_version = client.get_latest_versions('churn', stages=["None"])[0]

new_version = latest_version.version
new_run_id = latest_version.run_id

# Get accuracy of new model
new_acc = client.get_run(new_run_id).data.metrics["accuracy"]

# Get current Production model
prod_versions = client.get_latest_versions('churn', stages=["Production"])

if len(prod_versions) == 0:
    print("No production model found. Promoting new model.")

    client.transition_model_version_stage(
        name='churn',
        version=new_version,
        stage="Production"
    )

else:
    prod_version = prod_versions[0]
    prod_run_id = prod_version.run_id

    prod_acc = client.get_run(prod_run_id).data.metrics["accuracy"]

    print(f"New Accuracy: {new_acc}")
    print(f"Production Accuracy: {prod_acc}")

    # Compare and promote
    if new_acc > prod_acc:
        print("New model is better. Promoting to Production.")

        client.transition_model_version_stage(
            name='churn',
            version=new_version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("New model is worse. Keeping current Production model.")
