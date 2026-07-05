import os
import math
import cv2
import numpy as np


class ClusterVisualizer:

    def __init__(self, output_dir="../output",
                 save_dir="../reports/cluster_gallery"):

        self.output_dir = output_dir
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def create_gallery(self):

        clusters = sorted(os.listdir(self.output_dir))

        for cluster in clusters:

            cluster_path = os.path.join(self.output_dir, cluster)

            images = []

            for file in sorted(os.listdir(cluster_path)):

                img = cv2.imread(os.path.join(cluster_path, file))

                if img is None:
                    continue

                img = cv2.resize(img, (150, 150))

                images.append(img)

            if len(images) == 0:
                continue

            cols = 3
            rows = math.ceil(len(images) / cols)

            canvas = np.ones(
                (rows * 150, cols * 150, 3),
                dtype=np.uint8
            ) * 255

            for i, img in enumerate(images):

                r = i // cols
                c = i % cols

                canvas[
                    r*150:(r+1)*150,
                    c*150:(c+1)*150
                ] = img

            cv2.putText(
                canvas,
                cluster,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            save_path = os.path.join(
                self.save_dir,
                f"{cluster}.png"
            )

            cv2.imwrite(save_path, canvas)

        print("\nImages Generated Successfully!")