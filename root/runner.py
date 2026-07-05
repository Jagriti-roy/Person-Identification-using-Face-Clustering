import os
from setting import *
import numpy as np
from embeddings import FaceEmbedder
from clustering import PersonCluster
from confidence_calculator import ConfidenceCalculator
from sklearn.preprocessing import normalize
from evaluator import Evaluator
from report import ReportGenerator
from printer import header
from gallery import ClusterVisualizer
from website import *

embedder = FaceEmbedder()

embeddings = []
filenames = []


dataset = DATASET_PATH

header("Face Embeddings Generating ...")

for image_name in os.listdir(dataset):

    image_path = os.path.join(dataset, image_name)

    embedding = embedder.get_embedding(image_path)

    if embedding is None:
        print(f"{image_name} -> Face is not found")
        continue

    embeddings.append(embedding)
    filenames.append(image_name)

    print(f"{image_name} -> Image Emb Shape -> {embedding.shape}")

embeddings = np.array(embeddings)
embeddings = normalize(embeddings, norm="l2")
np.save(EMBEDDINGS_FILE, embeddings)
np.save(FILENAMES_FILE, filenames)

print("\nEmbeddings!")
print("Matrix Shape:", embeddings.shape)

cluster = PersonCluster()

labels = cluster.cluster_faces()

calculator = ConfidenceCalculator(
    embeddings,
    filenames,
    labels
)

calculator.calculate()
evaluator = Evaluator(
    filenames,
    labels
)

metrics = evaluator.evaluate()

report = ReportGenerator(
    embeddings,
    labels,
    "../confidence_scores.csv",
    metrics
)

report.generate()

visualizer = ClusterVisualizer()

visualizer.create_gallery()

dashboard = Dashboard()
dashboard.generate()