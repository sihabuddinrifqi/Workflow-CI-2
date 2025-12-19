import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():
    # Load dataset
    df = pd.read_csv("dataset_preprocessing.csv")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Set experiment
    mlflow.set_experiment("RFM_Clustering_Skilled")

    param_grid = [2, 3, 4, 5]

    best_score = -1

    for n_clusters in param_grid:
        with mlflow.start_run():
            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                max_iter=300,
                random_state=42
            )

            model.fit(X_scaled)
            labels = model.predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)

            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("silhouette_score", sil_score)

            mlflow.sklearn.log_model(model, artifact_path="model")

            if sil_score > best_score:
                best_score = sil_score

            print(f"n_clusters={n_clusters}, silhouette={sil_score:.4f}")

    print("Training selesai")


if __name__ == "__main__":
    main()
