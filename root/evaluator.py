import os
import numpy as np

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


class Evaluator:

    def __init__(self, filenames, predicted_labels):

        self.filenames = filenames
        self.predicted = predicted_labels

    def get_true_labels(self):

        labels = []

        label_map = {}

        current = 0

        for file in self.filenames:

            # Example:
            # Tom_Hanks_Tom_Hanks_0001.jpg

            parts = file.split("_")

            if parts[0] == "person":
                identity = "_".join(parts[:2])   # person_01
            else:
                identity = "_".join(parts[:-2])  # LFW names

            if identity not in label_map:
                label_map[identity] = current
                current += 1

            labels.append(label_map[identity])

        return np.array(labels)

    def evaluate(self):

        true_labels = self.get_true_labels()

        print("\n==============================")
        print(" CLUSTERING EVALUATION ")
        print("==============================")

        print(f"Images : {len(true_labels)}")
        print(f"Identities : {len(np.unique(true_labels))}")
        print(f"Clusters : {len(np.unique(self.predicted))}")

        print()

        print("Adjusted Rand Index :",
              round(adjusted_rand_score(true_labels, self.predicted),4))

        print("Normalized Mutual Information :",
              round(normalized_mutual_info_score(true_labels, self.predicted),4))

        print("Homogeneity :",
              round(homogeneity_score(true_labels, self.predicted),4))

        print("Completeness :",
              round(completeness_score(true_labels, self.predicted),4))

        print("V Measure :",
              round(v_measure_score(true_labels, self.predicted),4))
        
        return {
            "ARI": adjusted_rand_score(true_labels, self.predicted),
            "NMI": normalized_mutual_info_score(true_labels, self.predicted),
            "Homogeneity": homogeneity_score(true_labels, self.predicted),
            "Completeness": completeness_score(true_labels, self.predicted),
            "VMeasure": v_measure_score(true_labels, self.predicted),
        }