import os
import glob
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.types as T
from tqdm import tqdm

# --------------------------
# Start Spark session
# --------------------------
spark = SparkSession.builder \
    .appName("TNG300_DMParquetWrite_PandasToSpark") \
    .master("spark://sohnic:7077") \
    .config("spark.driver.memory", "100g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "30") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.conf.set("spark.sql.files.maxPartitionBytes", 64 * 1024 * 1024)  # 64 MB

# --------------------------
# Define schema
# --------------------------
dm_schema = T.StructType([
    T.StructField('px', T.FloatType(), True),
    T.StructField('py', T.FloatType(), True),
    T.StructField('pz', T.FloatType(), True),
    T.StructField('vx', T.FloatType(), True),
    T.StructField('vy', T.FloatType(), True),
    T.StructField('vz', T.FloatType(), True),
    T.StructField('id', T.LongType(), True)
])

# --------------------------
# Batch processing setup
# --------------------------
csv_dir = "/mnt/data1/TNG300/snap_099sorted"
pattern = os.path.join(csv_dir, "snap099_dm_sorted_x*_y*_z*.csv")
all_csv_files = sorted(glob.glob(pattern))
print(f"📁 Found {len(all_csv_files)} DM CSV files")

batch_size = 1000
n_batches = (len(all_csv_files) + batch_size - 1) // batch_size
parquet_output_dir = "hdfs://sohnic:54310/data/TNG300/snap_099/dm_particle_batches"

# --------------------------
# Process each batch
# --------------------------
for i in range(n_batches):
    batch_files = all_csv_files[i * batch_size : (i + 1) * batch_size]
    print(f"🌀 Processing batch {i+1}/{n_batches} with {len(batch_files)} files")

    # Read and concatenate using pandas
    try:
        df_list = [pd.read_csv(f) for f in batch_files if os.path.isfile(f)]
        if len(df_list) == 0:
            print("⚠️ No valid CSVs in this batch. Skipping.")
            continue
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"❌ Failed to read batch {i+1}: {e}")
        continue

    # Convert to Spark DataFrame
    try:
        spark_df = spark.createDataFrame(df, schema=dm_schema)
    except Exception as e:
        print(f"❌ Failed to convert batch {i+1} to Spark DF: {e}")
        continue

    # Save as Parquet
    out_path = os.path.join(parquet_output_dir, f"batch_{i:04d}.parquet")
    try:
        spark_df.repartition(50).write.option("compression", "snappy").mode("overwrite").save(out_path)
        print(f"✅ Saved batch {i+1} to {out_path}")
    except Exception as e:
        print(f"❌ Failed to write batch {i+1}: {e}")

# --------------------------
# Done
# --------------------------
spark.stop()
print("🚀 All batches processed.")
