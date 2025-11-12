import numpy as np
import pandas as pd
import glob
from pyspark.sql import SparkSession
import pyspark.sql.types as T
from tqdm import tqdm

# Start Spark session
spark = SparkSession.builder \
    .appName("TNG300_BatchedParquetWrite") \
    .master("spark://sohnic:7077") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()



# # Star particles
# # Schema for star particles
# schema = T.StructType([
#     T.StructField('px', T.FloatType(), True),
#     T.StructField('py', T.FloatType(), True),
#     T.StructField('pz', T.FloatType(), True),
#     T.StructField('vx', T.FloatType(), True),
#     T.StructField('vy', T.FloatType(), True),
#     T.StructField('vz', T.FloatType(), True),
#     T.StructField('mass', T.FloatType(), True),
#     T.StructField('id', T.LongType(), True),
#     T.StructField('formation_time', T.FloatType(), True),
# ])
# # List all CSVs
# ptl_files = np.sort(glob.glob('/mnt/data1/TNG300/snap_099sorted/snap099_star_sorted*.csv'))

# # Write in 10 batches
# n_batches = 10 # recommend to set it same as the number you use to divide the data cube for sorting particles
# batch_size = len(ptl_files) // n_batches

# for batch_idx in range(n_batches):
#     print(f"Processing batch {batch_idx + 1} / {n_batches} ...")
#     batch_files = ptl_files[batch_idx * batch_size : min((batch_idx + 1) * batch_size, len(ptl_files))]
#     # Initialize empty spark DataFrame
#     sparkdf = spark.createDataFrame([], schema)

#     for file in tqdm(batch_files, desc=f"Batch {batch_idx}", leave=False):
#         df = pd.read_csv(file)
#         tempdf = spark.createDataFrame(df[['px', 'py', 'pz', 'vx', 'vy', 'vz', 'mass', 'ID', 'Formation_time']], schema)
#         sparkdf = sparkdf.union(tempdf)

#     # Save the batch
#     outfile = f"hdfs://sohnic:54310/data/TNG300/snap_099/star_particle_batch{batch_idx}.parquet.snappy"
#     sparkdf.write.option("compression", "snappy").mode("overwrite").save(outfile)
#     print(f"✔️ Saved batch {batch_idx} to {outfile}")

# DM particles
# Schema for star particles
schema = T.StructType([
    T.StructField('px', T.FloatType(), True),
    T.StructField('py', T.FloatType(), True),
    T.StructField('pz', T.FloatType(), True),
    T.StructField('vx', T.FloatType(), True),
    T.StructField('vy', T.FloatType(), True),
    T.StructField('vz', T.FloatType(), True),
    T.StructField('id', T.LongType(), True)
])
# List all CSVs
ptl_files = np.sort(glob.glob('/mnt/data1/TNG300/snap_099sorted/snap099_dm_sorted*.csv'))

# Write in 10 batches
n_batches = 1000
batch_size = len(ptl_files) // n_batches

for batch_idx in range(n_batches):
    print(f"Processing batch {batch_idx + 1} / {n_batches} ...")
    batch_files = ptl_files[batch_idx * batch_size : min((batch_idx + 1) * batch_size, len(ptl_files))]
    # Initialize empty spark DataFrame
    sparkdf = spark.createDataFrame([], schema)

    for file in tqdm(batch_files, desc=f"Batch {batch_idx}", leave=False):
        df = pd.read_csv(file)
        tempdf = spark.createDataFrame(df[['px', 'py', 'pz', 'vx', 'vy', 'vz', 'ID']], schema)
        sparkdf = sparkdf.union(tempdf)

    # Save the batch
    outfile = f"hdfs://sohnic:54310/data/TNG300/snap_099/dm_particle_batch{batch_idx}.parquet.snappy"
    sparkdf = sparkdf.repartition(200)
    sparkdf.write.option("compression", "snappy").mode("overwrite").save(outfile)
    print(f"✔️ Saved batch {batch_idx} to {outfile}")
