import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ConfidenceCalculator:

    def __init__(self, embeddings, filenames, labels):
        self.embeddings = embeddings
        self.filenames = filenames
        self.labels = labels

    def calculate(self):

        results = []

        unique_labels = np.unique(self.labels)

        for label in unique_labels:

            indices = np.where(self.labels == label)[0]

            cluster_embeddings = self.embeddings[indices]

            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

            for idx in indices:

                similarity = cosine_similarity(
                    self.embeddings[idx].reshape(1, -1),
                    centroid
                )[0][0]

                confidence = similarity * 100

                results.append([
                    self.filenames[idx],
                    int(label),
                    round(confidence, 2)
                ])

        df = pd.DataFrame(
            results,
            columns=["Image", "Cluster", "Confidence (%)"]
        )

        df.to_csv("../confidence_scores.csv", index=False)

        print("\nConfidence scores saved successfully!")

        print(df)