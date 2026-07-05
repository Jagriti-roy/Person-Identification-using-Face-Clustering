import os
import pandas as pd
import numpy as np


class ReportGenerator:

    def __init__(self,
                 embeddings,
                 labels,
                 confidence_file,
                 evaluation):

        self.embeddings = embeddings
        self.labels = labels
        self.confidence_file = confidence_file
        self.evaluation = evaluation

    def generate(self):

        df = pd.read_csv(self.confidence_file)

        avg_conf = df["Confidence (%)"].mean()

        report = f"""
    ==========================================================
                PERSON GROUPING PERFORMANCE REPORT
    ==========================================================

    Embedding Length      : {self.embeddings.shape[1]}

    Processed Images      : {self.embeddings.shape[0]}

    Discovered Groups     : {len(np.unique(self.labels))}

    Average Group Score   : {avg_conf:.2f} %

    ----------------------------------------------------------

    Adjusted Rand Score   : {self.evaluation["ARI"]:.4f}

    Mutual Information    : {self.evaluation["NMI"]:.4f}

    Homogeneity Score     : {self.evaluation["Homogeneity"]:.4f}

    Completeness Score    : {self.evaluation["Completeness"]:.4f}

    Overall V-Measure     : {self.evaluation["VMeasure"]:.4f}

    ==========================================================
    """
        
        os.makedirs("../reports", exist_ok=True)

        with open("../reports/summary.txt", "w") as f:
            f.write(report)

        metrics = pd.DataFrame({

            "Metric":[

                "Embedding Dimension",

                "Images Processed",

                "Clusters Generated",

                "Average Confidence",

                "Adjusted Rand Index",

                "NMI",

                "Homogeneity",

                "Completeness",

                "V Measure"

            ],

            "Value":[

                self.embeddings.shape[1],

                self.embeddings.shape[0],

                len(np.unique(self.labels)),

                round(avg_conf,2),

                round(self.evaluation["ARI"],4),

                round(self.evaluation["NMI"],4),

                round(self.evaluation["Homogeneity"],4),

                round(self.evaluation["Completeness"],4),

                round(self.evaluation["VMeasure"],4)

            ]

        })

        metrics.to_csv(

            "../reports/metrics.csv",

            index=False

        )

        print("Metrics Generated!")

        print("\nSummary Generated!")