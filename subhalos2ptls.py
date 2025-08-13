import numpy as np
import pandas as pd
import glob
import sys
import h5py

from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import pyarrow as pa
import pyarrow.parquet as pqw

from functools import reduce
import operator
import gc

h = 0.6774
a = 1/(1+0.62)
box_size = 205000
t_h = 7.786*1e9

from pyspark import SparkContext   
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast, col, sqrt, pow, floor, monotonically_increasing_id, abs, pmod, least, row_number
import pyspark.sql.types as T
from pyspark import Row
from pyspark.sql.window import Window as W

spark = SparkSession.builder \
    .appName("MyApp") \
    .master("spark://sohnic:7077") \
    .config("spark.executor.memory", "100g")\
    .config("spark.driver.memory", "100g") \
    .getOrCreate()

sc = spark.sparkContext
sc.setCheckpointDir("hdfs://sohnic:54310/tmp/checkpoints")

spark.conf.set("spark.sql.debug.maxToStringFields", 500)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.hive.filesourcePartitionFileCacheSize", 524288000) # 500MB 

particle_file = 'hdfs://sohnic:54310/data/TNG300/snap99/snap099_cubic_indexed.parquet.snappy'
ptldf = spark.read.option("header","true").option("recursiveFileLookup","true").parquet(particle_file)

subhalofile = 'hdfs://sohnic:54310/data/TNG300/snap99/parquet/subhalo.parquet.snappy'
subdf = spark.read.option("header","true").option("recursiveFileLookup","true").parquet(subhalofile)

subdf = subdf.withColumnRenamed("px", "subhalo_px")
subdf = subdf.withColumnRenamed("py", "subhalo_py")
subdf = subdf.withColumnRenamed("pz", "subhalo_pz")
subdf = subdf.withColumnRenamed("vx", "subhalo_vx")
subdf = subdf.withColumnRenamed("vy", "subhalo_vy")
subdf = subdf.withColumnRenamed("vz", "subhalo_vz")
subdf = subdf.withColumnRenamed("StarHalfMass", "subhalo_StarHalfMass")
subdf = subdf.withColumnRenamed("sub_id", "subhalo_id")
subdf = subdf.filter(F.col("StarHalfMass")/h>0.01)

id_size = 6000
subbox_size = box_size/id_size
half_box = box_size/2

ptldf = ptldf.withColumn("ix", floor(F.col("px") / subbox_size ).cast('int'))
ptldf = ptldf.withColumn("iy", floor(F.col("py") / subbox_size ).cast('int'))
ptldf = ptldf.withColumn("iz", floor(F.col("pz") / subbox_size ).cast('int'))

subdf = subdf.withColumn("subhalo_ix", floor(F.col("subhalo_px") / subbox_size).cast('int'))
subdf = subdf.withColumn("subhalo_iy", floor(F.col("subhalo_py") / subbox_size).cast('int'))
subdf = subdf.withColumn("subhalo_iz", floor(F.col("subhalo_pz") / subbox_size).cast('int'))

broadcast_subdf = F.broadcast(subdf)

def int_ptl2subhalo(ptl_boxnumber, subhalo_boxnumber):
    id_diff = F.least(F.abs(F.col(ptl_boxnumber)-F.col(subhalo_boxnumber)), id_size-F.abs(F.col(ptl_boxnumber)-F.col(subhalo_boxnumber)))
    return id_diff <= 1

boxnumber_joined_df = (
    ptldf.alias("particles").join(
        broadcast_subdf.alias("subhalos"),
        (int_ptl2subhalo("particles.ix", "subhalos.subhalo_ix")  &
         int_ptl2subhalo("particles.iy", "subhalos.subhalo_iy")  &
         int_ptl2subhalo("particles.iz", "subhalos.subhalo_iz"))
        )
)

boxnumber_joined_df = boxnumber_joined_df.withColumn("rel_px", F.pmod(F.col("particles.px") - F.col("subhalos.subhalo_px") + half_box, box_size) - half_box)
boxnumber_joined_df = boxnumber_joined_df.withColumn("rel_py", F.pmod(F.col("particles.py") - F.col("subhalos.subhalo_py") + half_box, box_size) - half_box)
boxnumber_joined_df = boxnumber_joined_df.withColumn("rel_pz", F.pmod(F.col("particles.pz") - F.col("subhalos.subhalo_pz") + half_box, box_size) - half_box)

boxnumber_joined_df = boxnumber_joined_df.withColumn("sq_dist_subhalo2ptl", F.pow(F.col("rel_px"), 2) + F.pow(F.col("rel_py"), 2) + F.pow(F.col("rel_pz"), 2))

sq_radius = (50*h)**2
joined_df = boxnumber_joined_df.filter(F.col("sq_dist_subhalo2ptl")<sq_radius)

# connect particle data to subhalo catalog
subcat = pd.read_csv("subhalocat300.txt", sep=' ')
subcat_type =  T.StructType([T.StructField('subhalo_id',T.IntegerType(), True),
                             T.StructField('StarHalfRad',T.IntegerType(), True)
                      ])
subcat = spark.createDataFrame(subcat[["SubfindID", "StarHalfRad"]], subcat_type)
joined_df = joined_df.join(subcat, "subhalo_id")

# save the haddop dataframe
filename = 'hdfs://sohnic:54310/data/TNG300/snap99/subhalos2ptls.parquet.snappy'
joined_df.write.option("compression", "snappy").mode("overwrite").partitionBy("subhalo_id").parquet(filename)
