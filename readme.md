# System Requirements

1. python 3.7
2. Java 8

# Package requirements

pip3 install -r requirements.txt

# API Documentation

1. Compute Distance Matrix

compute_distance_matrix(df: DataFrame, id_column: str, distance_udf: F.udf)

Example:
euclidean_udf = F.udf(lambda x, y: float(distance.euclidean(x, y)), FloatType())
distance_matrix = compute_distance_matrix(sparkDF, "id", euclidean_udf)

2. Find distances for an id
find_distances(1, distance_matrix)

3. Find distance between two ids
find_distance(1, 2, distance_matrix)
