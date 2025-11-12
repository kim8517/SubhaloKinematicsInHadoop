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



# Star particles
schema = T.StructType([
    T.StructField('px', T.FloatType(), True),
    T.StructField('py', T.FloatType(), True),
    T.StructField('pz', T.FloatType(), True),
    T.StructField('vx', T.FloatType(), True),
    T.StructField('vy', T.FloatType(), True),
    T.StructField('vz', T.FloatType(), True),
    T.StructField('mass', T.FloatType(), True),
    T.StructField('ID', T.LongType(), True),
    T.StructField('Formation_time', T.FloatType(), True)
])
# construct the spark df
files = glob.glob('/mnt/data1/TNG300/snap_099sorted/snap099_star_sorted*.csv')
files = [f"file://{x}" for x in files]
print(files[:10])
sparkdf = spark.read.csv(
    files,
    header=True,
    schema=schema,
)

print(f"Number of rows loaded: {sparkdf.count()}")
sparkdf.show(5)

# Save the spark df as parquet
outfile = f"hdfs://sohnic:54310/data/TNG300/snap_099/star_particle.parquet"
sparkdf.write.option("compression", "snappy").mode("overwrite").save(outfile)
print(f"Saved star particles to {outfile}")

# DM particles
# construct the spark df
schema = T.StructType([
    T.StructField('px', T.FloatType(), True),
    T.StructField('py', T.FloatType(), True),
    T.StructField('pz', T.FloatType(), True),
    T.StructField('vx', T.FloatType(), True),
    T.StructField('vy', T.FloatType(), True),
    T.StructField('vz', T.FloatType(), True),
    T.StructField('ID', T.LongType(), True)
])
sparkdf = spark.read.csv(
    'file:///mnt/data1/TNG300/snap_099sorted/snap099_dm_sorted*.csv',
    header=True,
    schema=schema
)

# Save the spark df as parquet
outfile = f"hdfs://sohnic:54310/data/TNG300/snap_099/dm_particle.parquet"
sparkdf.write.option("compression", "snappy").mode("overwrite").save(outfile)
print(f"Saved dm particles to {outfile}")