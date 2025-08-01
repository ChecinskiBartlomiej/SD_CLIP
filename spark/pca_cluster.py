import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, regexp_replace, udf, row_number
from pyspark.ml.functions import vector_to_array
from paths.paths import embeddings_dir, results_dir
import math
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

def run_clustering(spark: SparkSession):

    df = spark.read.parquet(str(embeddings_dir / "clip_embeddings.parquet"))

    feat_cols = [c for c in df.columns if c.startswith("emb_")]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
    df_feat = assembler.transform(df)

    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    df_pca = pca.fit(df_feat).transform(df_feat)

    km = KMeans(k=3, featuresCol="pca_features", predictionCol="cluster").fit(df_pca)
    df_clustered = km.transform(df_pca)
    df_clustered = (
        df_clustered
        .withColumn("pca_array", vector_to_array(col("pca_features")))
        .withColumn("pca1", col("pca_array")[0])
        .withColumn("pca2", col("pca_array")[1])
        .withColumn("prompt", regexp_replace(col("prompt"), "_", " "))
    )

    centers = km.clusterCenters() 
    center_list = [list(c) for c in centers]
    bc_centers = spark.sparkContext.broadcast(center_list)


    dist_udf = udf(
        lambda cluster_id, coords: float(
            math.sqrt(
                sum(
                    (coords[i] - bc_centers.value[cluster_id][i])**2
                    for i in range(len(bc_centers.value[cluster_id]))
                )
            )
        ),
        DoubleType()
    )

    df_with_dist = df_clustered.withColumn("distance", dist_udf(col("cluster"), col("pca_array")))

    window = Window.partitionBy("cluster").orderBy(col("distance").desc())
    df_furthest = df_with_dist \
        .withColumn("rank", row_number().over(window)) \
        .filter(col("rank") == 1) \
        .select("cluster", "filename", "prompt", "distance", "pca1", "pca2")

    df_furthest.write.mode("overwrite") \
        .partitionBy("cluster") \
        .parquet(str(results_dir / "furthest"))
    
    df_clustered.write.mode("overwrite") \
        .partitionBy("cluster") \
        .parquet(str(results_dir / "clustered"))
    


