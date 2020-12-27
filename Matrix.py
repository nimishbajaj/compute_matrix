import time

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.random import RandomRDDs
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import FloatType
from scipy.spatial import distance
import pyspark.sql.functions as F


def compute_distance_matrix(df: DataFrame, id_column: str, distance_udf: F.udf):
    # select the columns on which the distance needs to be found
    vector_columns = df.columns
    vector_columns.remove(id_column)

    # vectorize the two dataframes
    assembler = VectorAssembler(inputCols=vector_columns, outputCol="features")
    source_vectorized = assembler.transform(df).select(id_column, assembler.getOutputCol()).alias("source")
    destination_vectorized = assembler.transform(df).select(id_column, assembler.getOutputCol()).alias("destination")

    cross_df = source_vectorized.crossJoin(destination_vectorized)
    return cross_df \
        .withColumn("distance", distance_udf(F.col("source." + assembler.getOutputCol()),
                                             F.col("destination." + assembler.getOutputCol()))) \
        .select("source." + id_column, "destination." + id_column, "distance").persist()


def generate_spark_matrix(nrows: int, ncols: int, spark):
    df = RandomRDDs.uniformVectorRDD(spark.sparkContext, nrows, ncols).map(lambda a: a.tolist()).toDF().persist()
    return df


def find_distances(source_id: int, distance_matrix: DataFrame):
    return distance_matrix.filter(F.col("source." + id_column) == source_id)


def find_distance(source_id: int, destination_id: int, distance_matrix: DataFrame):
    return distance_matrix \
        .filter(F.col("source." + id_column) == source_id) \
        .filter(F.col("destination." + id_column) == destination_id)


if __name__ == "__main__":
    start = time.time()
    spark = SparkSession.builder.appName("distance_matrix").master("local[*]").getOrCreate()

    # set id_column of the dataframe
    id_column = "id"

    # generate dataset with random values for testing the script
    sparkDF: DataFrame = generate_spark_matrix(100, 1, spark)
    sparkDF = sparkDF.withColumn(id_column, monotonically_increasing_id())

    # Define the UDF to compute the distance between vectors
    euclidean_udf = F.udf(lambda x, y: float(distance.euclidean(x, y)), FloatType())

    # API1 - Find the distance matrix
    distance_matrix = compute_distance_matrix(sparkDF, id_column, euclidean_udf)

    # API2 - Find distances for an id
    find_distances(1, distance_matrix).show()

    # API3 - Find distance between two ids
    find_distance(1, 2, distance_matrix).show()

    end = time.time()
    print("Time elapsed:", end - start, "seconds")
