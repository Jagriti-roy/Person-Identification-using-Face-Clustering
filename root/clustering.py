import os
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from setting import *

class PersonCluster:

    def __init__(self):
        self.embeddings = np.load(EMBEDDINGS_FILE)
        
        self.filenames = np.load(FILENAMES_FILE, allow_pickle=True)

    def cluster_faces(self):

        clustering = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=MIN_SAMPLES,
            metric=COSINE_METRIC
        )

        labels = clustering.fit_predict(self.embeddings)

        print("\nCluster Labels:")
        print(labels)

        output_dir = OUTPUT_PATH

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        for filename, label in zip(self.filenames, labels):

            cluster_folder = os.path.join(
                output_dir,
                f"cluster_{label}"
            )

            os.makedirs(cluster_folder, exist_ok=True)

            src = os.path.join("../data", filename)

            dst = os.path.join(cluster_folder, filename)

            shutil.copy(src, dst)

        print("\nImages copied successfully!")

        return labels