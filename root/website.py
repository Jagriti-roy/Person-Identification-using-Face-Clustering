import pandas as pd


class Dashboard:

    def __init__(self):

        self.metrics = "../reports/metrics.csv"
        self.confidence = "../confidence_scores.csv"

    def generate(self):

        metrics = pd.read_csv(self.metrics)

        confidence = pd.read_csv(self.confidence)

        avg_conf = confidence["Confidence (%)"].mean()

        cards = ""

        for _, row in metrics.iterrows():

            icon = "📊"

            if "Image" in row["Metric"]:
                icon = "📸"

            elif "Cluster" in row["Metric"]:
                icon = "📂"

            elif "Confidence" in row["Metric"]:
                icon = "🎯"

            elif "Rand" in row["Metric"]:
                icon = "🧠"

            elif "NMI" in row["Metric"]:
                icon = "📈"

            elif "Homogeneity" in row["Metric"]:
                icon = "🧩"

            elif "Completeness" in row["Metric"]:
                icon = "✅"

            cards += f"""
            <div class="card">
                <div class="icon">{icon}</div>
                <div class="metric">{row['Metric']}</div>
                <div class="value">{row['Value']}</div>
            </div>
            """

        import os

        gallery = ""

        gallery_path = "../reports/cluster_gallery"

        if os.path.exists(gallery_path):

            images = sorted(os.listdir(gallery_path))

            for image in images:

                gallery += f"""

                <div class="gallery-card">

                    <h3>{image.replace('.png','').replace('_',' ').title()}</h3>

                    <img src="cluster_gallery/{image}">

                </div>

                """

        html = f"""
        <!DOCTYPE html>

        <html>

        <head>

        <meta charset="utf-8">

        <title>Person Identification Analytics</title>

        <style>

        .container{{
        max-width:1250px;
        margin:auto;
        }}

        .header{{
        background:#2563eb;
        color:white;
        padding:25px;
        border-radius:10px;
        text-align:center;
        margin-bottom:30px;
        }}

        .header h1{{
        font-size:34px;
        margin-bottom:8px;
        }}

        .header p{{
        font-size:17px;
        opacity:.9;
        }}

        .section{{
        background:white;
        padding:25px;
        border-radius:10px;
        box-shadow:0 2px 10px rgba(0,0,0,.08);
        margin-bottom:35px;
        }}

        .section h2{{
        margin-bottom:20px;
        color:#2563eb;
        font-size:24px;
        }}

        .gallery{{
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
        gap:20px;
        }}

        .gallery-card{{
        background:#fafafa;
        padding:15px;
        border:1px solid #ddd;
        border-radius:10px;
        text-align:center;
        }}

        .gallery-card img{{
        width:100%;
        border-radius:8px;
        }}

        .gallery-card h3{{
        margin-bottom:12px;
        color:#333;
        }}


        .footer{{
        margin-top:40px;
        text-align:center;
        font-size:14px;
        color:#777;
        }}

        </style>

        </head>

        <body>

        <div class="container">

        <div class="header">

        <h1>Person Identification Analytics</h1>


        </div>

        <div class="section">

        <h2>Cluster Gallery</h2>

        <div class="gallery">

        {gallery}

        </div>

        </div>


        </div>

        </body>

        </html>

        """

        with open("../reports/index.html", "w", encoding="utf-8") as f:

            f.write(html)

        print("\nWebsite Successfully!")