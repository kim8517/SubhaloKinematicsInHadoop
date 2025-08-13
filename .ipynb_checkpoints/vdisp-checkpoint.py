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
spark.conf.set("spark.sql.hive.filesourcePartitionFileCacheSize", 4294967296)  # 4GB

# particle data
filename = 'hdfs://sohnic:54310/data/TNG300/snap99/parquet/subhalos2ptls.parquet.snappy'
df = spark.read.parquet(filename)

from pyspark.sql import Window as W

# Define subhalo window for partitioning by "subhalo_id"
subhalo_window = W.partitionBy("subhalo_id")

# function to calculate 3D stellar rotation velocity and stellar velocity dispersion based on particles within aper
def rotv_vdisp_aperture(df, aperture):
    # filter a Spark Data Frame with d_{ptl-subhalo}<aperture.
    if aperture:
        distance_limit = (aperture * h) ** 2
        filtered_df = df.filter(F.col("sq_dist_subhalo2ptl") <= distance_limit)
    else:
        filtered_df = df

    # (0)
    filtered_df = filtered_df.withColumn("r", F.sqrt(F.col("rel_px")**2+F.col("rel_py")**2+F.col("rel_pz")**2))
    filtered_df = filtered_df.withColumn("unit_rel_px", F.col("rel_px")/F.col("r"))
    filtered_df = filtered_df.withColumn("unit_rel_py", F.col("rel_py")/F.col("r"))
    filtered_df = filtered_df.withColumn("unit_rel_pz", F.col("rel_pz")/F.col("r"))

    # (1)
    filtered_df = filtered_df.withColumn("mass_sum", F.sum("mass").over(subhalo_window)) # sum over each subhalo (the particle with the same subhalo_id)

    # (2)
    filtered_df = filtered_df.withColumn("vx_weighted", F.col("mass") * F.col("vx"))
    filtered_df = filtered_df.withColumn("vy_weighted", F.col("mass") * F.col("vy"))
    filtered_df = filtered_df.withColumn("vz_weighted", F.col("mass") * F.col("vz"))
    
    filtered_df = filtered_df.withColumn("vx_avg", F.sum("vx_weighted").over(subhalo_window) / F.col("mass_sum"))
    filtered_df = filtered_df.withColumn("vy_avg", F.sum("vy_weighted").over(subhalo_window) / F.col("mass_sum"))
    filtered_df = filtered_df.withColumn("vz_avg", F.sum("vz_weighted").over(subhalo_window) / F.col("mass_sum"))

    # (3)
    filtered_df = filtered_df.withColumn("rel_vx", F.col("vx") - F.col("vx_avg"))
    filtered_df = filtered_df.withColumn("rel_vy", F.col("vy") - F.col("vy_avg"))
    filtered_df = filtered_df.withColumn("rel_vz", F.col("vz") - F.col("vz_avg"))

    filtered_df = filtered_df.drop("vx_avg", "vy_avg", "vz_avg")

    # (4)
    filtered_df = filtered_df.withColumn("dispersion_x", F.col("rel_vx")**2)
    filtered_df = filtered_df.withColumn("dispersion_y", F.col("rel_vy")**2) 
    filtered_df = filtered_df.withColumn("dispersion_z", F.col("rel_vz")**2) 
    filtered_df = filtered_df.withColumn("dispersion_weighted", F.col("mass") * (F.col("dispersion_x") + F.col("dispersion_y") + F.col("dispersion_z")))
    
    # (5)
    rotv_vdisp_df = filtered_df.groupBy("subhalo_id").agg((F.sum("dispersion_weighted")*0.5).alias(f"total_kinetic_energy_{aperture}"))
    # .groupBy("a"): groupiong the spark DataFrame based on the column "a"
    # .agg(X): perfroms aggregate calculation on each group
    
    # (6)
    vdisp_x_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_x")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_x"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_x_df, "subhalo_id")
    vdisp_y_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_y")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_y"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_y_df, "subhalo_id")
    vdisp_z_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_z")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_z"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_z_df, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.withColumn(f"mass_weighted_velocity_dispersion_{aperture}",
                            F.col(f"mass_weighted_velocity_dispersion_{aperture}_x")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_y")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_z"))

    filtered_df = filtered_df.drop("dispersion_x", "dispersion_y", "dispersion_z", "dispersion_weighted")
    
    # (7)
    filtered_df = filtered_df.withColumn("j_x", F.col("mass") * ( F.col("rel_py") * F.col("rel_vz") - F.col("rel_pz") * F.col("rel_vy") ))
    filtered_df = filtered_df.withColumn("j_y", F.col("mass") * ( F.col("rel_pz") * F.col("rel_vx") - F.col("rel_px") * F.col("rel_vz") ))
    filtered_df = filtered_df.withColumn("j_z", F.col("mass") * ( F.col("rel_px") * F.col("rel_vy") - F.col("rel_py") * F.col("rel_vx") ))

    # (8)`
    filtered_df = filtered_df.withColumn("j_tot_x", F.sum("j_x").over(subhalo_window))
    filtered_df = filtered_df.withColumn("j_tot_y", F.sum("j_y").over(subhalo_window))
    filtered_df = filtered_df.withColumn("j_tot_z", F.sum("j_z").over(subhalo_window))

    # (9)
    filtered_df = filtered_df.withColumn("j_tot", F.sqrt(F.col("j_tot_x")*F.col("j_tot_x") + F.col("j_tot_y")*F.col("j_tot_y") + F.col("j_tot_z")*F.col("j_tot_z")))

    # (10)
    filtered_df = filtered_df.withColumn("j_rot", (F.col("j_x")*F.col("j_tot_x") + F.col("j_y")*F.col("j_tot_y") + F.col("j_z")*F.col("j_tot_z")) / F.col("j_tot"))

    filtered_df = filtered_df.drop("j_x", "j_y", "j_z")
    
    # (11)
    filtered_df = filtered_df.withColumn("R_tot", (F.col("rel_px")*F.col("j_tot_x") + F.col("rel_py")*F.col("j_tot_y") + F.col("rel_pz")*F.col("j_tot_z")) / F.col("j_tot"))
    
    # (12)
    filtered_df = filtered_df.withColumn("R_rot", F.sqrt( F.col("sq_dist_subhalo2ptl") - F.col("R_tot")*F.col("R_tot")))
    
    filtered_df = filtered_df.drop("R_tot")

    # (13)
    filtered_df = filtered_df.filter(F.col("R_rot") > 0)
    filtered_df = filtered_df.withColumn("mV_rot", F.col("j_rot")/F.col("R_rot"))
    
    filtered_df = filtered_df.drop("R_rot")

    # (14)
    filtered_df = filtered_df.withColumn("Krot", 0.5*F.col("mV_rot")**2/F.col("mass"))
    Krot_groupdf = filtered_df.groupBy("subhalo_id").agg((F.sum("Krot")).alias(f"rotation_kinetic_energy_{aperture}"))
    rotv_vdisp_df = rotv_vdisp_df.join(Krot_groupdf, "subhalo_id")
    
    filtered_df = filtered_df.drop("Krot")

    # (15)
    rotv_groupdf = filtered_df.groupBy("subhalo_id").agg((F.sum("mV_rot") / F.max("mass_sum")).alias(f"mass_weighted_rotation_velocity_{aperture}"))
    filtered_df = filtered_df.join(rotv_groupdf, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.join(rotv_groupdf, "subhalo_id")


    # (16)
    filtered_df = filtered_df.withColumn("unitJ_x", F.col("j_tot_x")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJ_y", F.col("j_tot_y")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJ_z", F.col("j_tot_z")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_x", F.col("unitJ_y")*F.col("rel_pz")-F.col("unitJ_z")*F.col("rel_py"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_y", F.col("unitJ_z")*F.col("rel_px")-F.col("unitJ_x")*F.col("rel_pz"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_z", F.col("unitJ_x")*F.col("rel_py")-F.col("unitJ_y")*F.col("rel_px"))

    filtered_df = filtered_df.drop("unitJ_x", "unitJ_y", "unitJ_z")

    # (17)
    filtered_df = filtered_df.withColumn("unitJcrossRvec",
                                         F.sqrt(F.col("unitJcrossRvec_x")**2+F.col("unitJcrossRvec_y")**2+F.col("unitJcrossRvec_z")**2))

    # (18)
    filtered_df = filtered_df.withColumn("unitPhi_x", F.col("unitJcrossRvec_x")/F.col("unitJcrossRvec"))
    filtered_df = filtered_df.withColumn("unitPhi_y", F.col("unitJcrossRvec_y")/F.col("unitJcrossRvec"))
    filtered_df = filtered_df.withColumn("unitPhi_z", F.col("unitJcrossRvec_z")/F.col("unitJcrossRvec"))

    filtered_df = filtered_df.drop("unitJcrossRvec_x", "unitJcrossRvec_y", "unitJcrossRvec_z", "unitJcrossRvec")

    # (19)
    filtered_df = filtered_df.withColumn("unitTheta_x", F.col("unitPhi_y")*F.col("unit_rel_pz")-F.col("unitPhi_z")*F.col("unit_rel_py"))
    filtered_df = filtered_df.withColumn("unitTheta_y", F.col("unitPhi_z")*F.col("unit_rel_px")-F.col("unitPhi_x")*F.col("unit_rel_pz"))
    filtered_df = filtered_df.withColumn("unitTheta_z", F.col("unitPhi_x")*F.col("unit_rel_py")-F.col("unitPhi_y")*F.col("unit_rel_px"))

    # (20)
    filtered_df = filtered_df.withColumn("rel_vr", F.col("rel_vx")*F.col("unit_rel_px")+F.col("rel_vy")*F.col("unit_rel_py")
                                         +F.col("rel_vz")*F.col("unit_rel_pz"))
    filtered_df = filtered_df.withColumn("rel_vphi", F.col("rel_vx")*F.col("unitPhi_x")+F.col("rel_vy")*F.col("unitPhi_y")
                                         +F.col("rel_vz")*F.col("unitPhi_z"))
    filtered_df = filtered_df.withColumn("rel_vtheta", F.col("rel_vx")*F.col("unitTheta_x")+F.col("rel_vy")*F.col("unitTheta_y")
                                         +F.col("rel_vz")*F.col("unitTheta_z"))

    filtered_df = filtered_df.drop("unit_rel_px", "unit_rel_py", "unit_rel_pz", "unitTheta_x", "unitTheta_y", "unitTheta_z")

    # (21)
    filtered_df = filtered_df.withColumn("dispersion_r", F.col("rel_vr")**2)
    filtered_df = filtered_df.withColumn("dispersion_phi", F.col("rel_vphi")**2)
    filtered_df = filtered_df.withColumn("dispersion_theta", F.col("rel_vtheta")**2)
    
    vdisp_r_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_r")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_r"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_r_df, "subhalo_id")
    vdisp_theta_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_theta")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_theta"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_theta_df, "subhalo_id")
    vdisp_phi_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_phi")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_phi"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_phi_df, "subhalo_id")

    filtered_df = filtered_df.drop("dispersion_r", "dispersion_phi", "dispersion_theta", "rel_vr", "rel_vtheta")

    # (22)
    vdisp_phi_no_rotation_df = filtered_df.groupBy("subhalo_id").agg((F.sum((F.col("rel_vphi")-F.col(f"mass_weighted_rotation_velocity_{aperture}"))**2*F.col("mass"))/ F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_phi_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_phi_no_rotation_df, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.withColumn(f"mass_weighted_velocity_dispersion_{aperture}_no_rotation",
                                             F.col(f"mass_weighted_velocity_dispersion_{aperture}_r")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_phi_no_rotation")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_theta"))
    
    filtered_df = filtered_df.drop("rel_vphi")
    
    # (23)
    filtered_df = filtered_df.withColumn("dispersion_x_no_rotation", (F.col("rel_vx")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_x"))**2)
    filtered_df = filtered_df.withColumn("dispersion_y_no_rotation", (F.col("rel_vy")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_y"))**2)
    filtered_df = filtered_df.withColumn("dispersion_z_no_rotation", (F.col("rel_vz")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_z"))**2)
    vdisp_x_df_no_rotation = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_x_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_x_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_x_df_no_rotation, "subhalo_id")
    vdisp_y_df_no_rotation = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_y_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_y_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_y_df_no_rotation, "subhalo_id")
    vdisp_z_df_no_rotation  = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_z_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_z_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_z_df_no_rotation, "subhalo_id")
    
    
    return rotv_vdisp_df

from pyspark.sql import Window as W

# Define subhalo window for partitioning by "subhalo_id"
subhalo_window = W.partitionBy("subhalo_id")

# function to calculate 3D stellar rotation velocity and stellar velocity dispersion based on particles within aper
def rotv_vdisp_column(df, column):
    filtered_df = df.filter(F.col("sq_dist_subhalo2ptl") <= (F.col(column)*h)**2)
    aperture = column

    # (0)
    filtered_df = filtered_df.withColumn("r", F.sqrt(F.col("rel_px")**2+F.col("rel_py")**2+F.col("rel_pz")**2))
    filtered_df = filtered_df.withColumn("unit_rel_px", F.col("rel_px")/F.col("r"))
    filtered_df = filtered_df.withColumn("unit_rel_py", F.col("rel_py")/F.col("r"))
    filtered_df = filtered_df.withColumn("unit_rel_pz", F.col("rel_pz")/F.col("r"))

    # (1)
    filtered_df = filtered_df.withColumn("mass_sum", F.sum("mass").over(subhalo_window)) # sum over each subhalo (the particle with the same subhalo_id)

    # (2)
    filtered_df = filtered_df.withColumn("vx_weighted", F.col("mass") * F.col("vx"))
    filtered_df = filtered_df.withColumn("vy_weighted", F.col("mass") * F.col("vy"))
    filtered_df = filtered_df.withColumn("vz_weighted", F.col("mass") * F.col("vz"))
    
    filtered_df = filtered_df.withColumn("vx_avg", F.sum("vx_weighted").over(subhalo_window) / F.col("mass_sum"))
    filtered_df = filtered_df.withColumn("vy_avg", F.sum("vy_weighted").over(subhalo_window) / F.col("mass_sum"))
    filtered_df = filtered_df.withColumn("vz_avg", F.sum("vz_weighted").over(subhalo_window) / F.col("mass_sum"))

    # (3)
    filtered_df = filtered_df.withColumn("rel_vx", F.col("vx") - F.col("vx_avg"))
    filtered_df = filtered_df.withColumn("rel_vy", F.col("vy") - F.col("vy_avg"))
    filtered_df = filtered_df.withColumn("rel_vz", F.col("vz") - F.col("vz_avg"))

    filtered_df = filtered_df.drop("vx_avg", "vy_avg", "vz_avg")

    # (4)
    filtered_df = filtered_df.withColumn("dispersion_x", F.col("rel_vx")**2)
    filtered_df = filtered_df.withColumn("dispersion_y", F.col("rel_vy")**2) 
    filtered_df = filtered_df.withColumn("dispersion_z", F.col("rel_vz")**2) 
    filtered_df = filtered_df.withColumn("dispersion_weighted", F.col("mass") * (F.col("dispersion_x") + F.col("dispersion_y") + F.col("dispersion_z")))
    
    # (5)
    rotv_vdisp_df = filtered_df.groupBy("subhalo_id").agg((F.sum("dispersion_weighted")*0.5).alias(f"total_kinetic_energy_{aperture}"))
    # .groupBy("a"): groupiong the spark DataFrame based on the column "a"
    # .agg(X): perfroms aggregate calculation on each group
    
    # (6)
    vdisp_x_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_x")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_x"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_x_df, "subhalo_id")
    vdisp_y_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_y")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_y"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_y_df, "subhalo_id")
    vdisp_z_df = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_z")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_z"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_z_df, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.withColumn(f"mass_weighted_velocity_dispersion_{aperture}",
                            F.col(f"mass_weighted_velocity_dispersion_{aperture}_x")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_y")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_z"))

    filtered_df = filtered_df.drop("dispersion_x", "dispersion_y", "dispersion_z", "dispersion_weighted")
    
    # (7)
    filtered_df = filtered_df.withColumn("j_x", F.col("mass") * ( F.col("rel_py") * F.col("rel_vz") - F.col("rel_pz") * F.col("rel_vy") ))
    filtered_df = filtered_df.withColumn("j_y", F.col("mass") * ( F.col("rel_pz") * F.col("rel_vx") - F.col("rel_px") * F.col("rel_vz") ))
    filtered_df = filtered_df.withColumn("j_z", F.col("mass") * ( F.col("rel_px") * F.col("rel_vy") - F.col("rel_py") * F.col("rel_vx") ))

    # (8)`
    filtered_df = filtered_df.withColumn("j_tot_x", F.sum("j_x").over(subhalo_window))
    filtered_df = filtered_df.withColumn("j_tot_y", F.sum("j_y").over(subhalo_window))
    filtered_df = filtered_df.withColumn("j_tot_z", F.sum("j_z").over(subhalo_window))

    # (9)
    filtered_df = filtered_df.withColumn("j_tot", F.sqrt(F.col("j_tot_x")*F.col("j_tot_x") + F.col("j_tot_y")*F.col("j_tot_y") + F.col("j_tot_z")*F.col("j_tot_z")))

    # (10)
    filtered_df = filtered_df.withColumn("j_rot", (F.col("j_x")*F.col("j_tot_x") + F.col("j_y")*F.col("j_tot_y") + F.col("j_z")*F.col("j_tot_z")) / F.col("j_tot"))

    filtered_df = filtered_df.drop("j_x", "j_y", "j_z")
    
    # (11)
    filtered_df = filtered_df.withColumn("R_tot", (F.col("rel_px")*F.col("j_tot_x") + F.col("rel_py")*F.col("j_tot_y") + F.col("rel_pz")*F.col("j_tot_z")) / F.col("j_tot"))
    
    # (12)
    filtered_df = filtered_df.withColumn("R_rot", F.sqrt( F.col("sq_dist_subhalo2ptl") - F.col("R_tot")*F.col("R_tot")))
    
    filtered_df = filtered_df.drop("R_tot")

    # (13)
    filtered_df = filtered_df.filter(F.col("R_rot") > 0)
    filtered_df = filtered_df.withColumn("mV_rot", F.col("j_rot")/F.col("R_rot"))
    
    filtered_df = filtered_df.drop("R_rot")

    # (14)
    filtered_df = filtered_df.withColumn("Krot", 0.5*F.col("mV_rot")**2/F.col("mass"))
    Krot_groupdf = filtered_df.groupBy("subhalo_id").agg((F.sum("Krot")).alias(f"rotation_kinetic_energy_{aperture}"))
    rotv_vdisp_df = rotv_vdisp_df.join(Krot_groupdf, "subhalo_id")
    
    filtered_df = filtered_df.drop("Krot")

    # (15)
    rotv_groupdf = filtered_df.groupBy("subhalo_id").agg((F.sum("mV_rot") / F.max("mass_sum")).alias(f"mass_weighted_rotation_velocity_{aperture}"))
    filtered_df = filtered_df.join(rotv_groupdf, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.join(rotv_groupdf, "subhalo_id")


    # (16)
    filtered_df = filtered_df.withColumn("unitJ_x", F.col("j_tot_x")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJ_y", F.col("j_tot_y")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJ_z", F.col("j_tot_z")/F.col("j_tot"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_x", F.col("unitJ_y")*F.col("rel_pz")-F.col("unitJ_z")*F.col("rel_py"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_y", F.col("unitJ_z")*F.col("rel_px")-F.col("unitJ_x")*F.col("rel_pz"))
    filtered_df = filtered_df.withColumn("unitJcrossRvec_z", F.col("unitJ_x")*F.col("rel_py")-F.col("unitJ_y")*F.col("rel_px"))

    filtered_df = filtered_df.drop("unitJ_x", "unitJ_y", "unitJ_z")

    # (17)
    filtered_df = filtered_df.withColumn("unitJcrossRvec",
                                         F.sqrt(F.col("unitJcrossRvec_x")**2+F.col("unitJcrossRvec_y")**2+F.col("unitJcrossRvec_z")**2))

    # (18)
    filtered_df = filtered_df.withColumn("unitPhi_x", F.col("unitJcrossRvec_x")/F.col("unitJcrossRvec"))
    filtered_df = filtered_df.withColumn("unitPhi_y", F.col("unitJcrossRvec_y")/F.col("unitJcrossRvec"))
    filtered_df = filtered_df.withColumn("unitPhi_z", F.col("unitJcrossRvec_z")/F.col("unitJcrossRvec"))

    filtered_df = filtered_df.drop("unitJcrossRvec_x", "unitJcrossRvec_y", "unitJcrossRvec_z", "unitJcrossRvec")

    # (19)
    filtered_df = filtered_df.withColumn("unitTheta_x", F.col("unitPhi_y")*F.col("unit_rel_pz")-F.col("unitPhi_z")*F.col("unit_rel_py"))
    filtered_df = filtered_df.withColumn("unitTheta_y", F.col("unitPhi_z")*F.col("unit_rel_px")-F.col("unitPhi_x")*F.col("unit_rel_pz"))
    filtered_df = filtered_df.withColumn("unitTheta_z", F.col("unitPhi_x")*F.col("unit_rel_py")-F.col("unitPhi_y")*F.col("unit_rel_px"))

    # (20)
    filtered_df = filtered_df.withColumn("rel_vr", F.col("rel_vx")*F.col("unit_rel_px")+F.col("rel_vy")*F.col("unit_rel_py")
                                         +F.col("rel_vz")*F.col("unit_rel_pz"))
    filtered_df = filtered_df.withColumn("rel_vphi", F.col("rel_vx")*F.col("unitPhi_x")+F.col("rel_vy")*F.col("unitPhi_y")
                                         +F.col("rel_vz")*F.col("unitPhi_z"))
    filtered_df = filtered_df.withColumn("rel_vtheta", F.col("rel_vx")*F.col("unitTheta_x")+F.col("rel_vy")*F.col("unitTheta_y")
                                         +F.col("rel_vz")*F.col("unitTheta_z"))

    filtered_df = filtered_df.drop("unit_rel_px", "unit_rel_py", "unit_rel_pz", "unitTheta_x", "unitTheta_y", "unitTheta_z")

    # (21)
    filtered_df = filtered_df.withColumn("dispersion_r", F.col("rel_vr")**2)
    filtered_df = filtered_df.withColumn("dispersion_phi", F.col("rel_vphi")**2)
    filtered_df = filtered_df.withColumn("dispersion_theta", F.col("rel_vtheta")**2)
    
    vdisp_r_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_r")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_r"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_r_df, "subhalo_id")
    vdisp_theta_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_theta")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_theta"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_theta_df, "subhalo_id")
    vdisp_phi_df = filtered_df.groupBy("subhalo_id").agg((F.sum(F.col("dispersion_phi")*F.col("mass")) / F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_phi"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_phi_df, "subhalo_id")

    filtered_df = filtered_df.drop("dispersion_r", "dispersion_phi", "dispersion_theta", "rel_vr", "rel_vtheta")

    # (22)
    vdisp_phi_no_rotation_df = filtered_df.groupBy("subhalo_id").agg((F.sum((F.col("rel_vphi")-F.col(f"mass_weighted_rotation_velocity_{aperture}"))**2*F.col("mass"))/ F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_phi_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_phi_no_rotation_df, "subhalo_id")
    rotv_vdisp_df = rotv_vdisp_df.withColumn(f"mass_weighted_velocity_dispersion_{aperture}_no_rotation",
                                             F.col(f"mass_weighted_velocity_dispersion_{aperture}_r")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_phi_no_rotation")+F.col(f"mass_weighted_velocity_dispersion_{aperture}_theta"))
    
    filtered_df = filtered_df.drop("rel_vphi")
    
    # (23)
    filtered_df = filtered_df.withColumn("dispersion_x_no_rotation", (F.col("rel_vx")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_x"))**2)
    filtered_df = filtered_df.withColumn("dispersion_y_no_rotation", (F.col("rel_vy")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_y"))**2)
    filtered_df = filtered_df.withColumn("dispersion_z_no_rotation", (F.col("rel_vz")-F.col(f"mass_weighted_rotation_velocity_{aperture}")*F.col("unitPhi_z"))**2)
    vdisp_x_df_no_rotation = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_x_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_x_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_x_df_no_rotation, "subhalo_id")
    vdisp_y_df_no_rotation = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_y_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_y_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_y_df_no_rotation, "subhalo_id")
    vdisp_z_df_no_rotation  = filtered_df.groupBy("subhalo_id").agg(
        (F.sum(F.col("dispersion_z_no_rotation")*F.col("mass"))/F.sum("mass")).alias(f"mass_weighted_velocity_dispersion_{aperture}_z_no_rotation"))
    rotv_vdisp_df = rotv_vdisp_df.join(vdisp_z_df_no_rotation, "subhalo_id")
    
    return rotv_vdisp_df

rotv_vdisp_df = rotv_vdisp_aperture(df, 50)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp50.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_aperture(df, 30)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp30.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_aperture(df, 20)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp20.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_aperture(df, 10)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp10.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_aperture(df, 5)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp5.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_aperture(df, 3)
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdisp3.txt', sep=' ', index=False)
rotv_vdisp_df = rotv_vdisp_column(df, "StarHalfRad")
rotv_vdisp_df.toPandas().sort_values(by="subhalo_id").to_csv('result/vdispStarHalfRad.txt', sep=' ', index=False)


