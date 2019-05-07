import numpy as np
import h5py
from pyspark import SparkConf, SparkContext
import sys

# NOTE: This code requires you to have downloaded simulation snapshots
# to an EBS storage attached to your cluster.

# If you are interested in getting access to the Caterpillar particle data,
# please contact the Caterpillar team / email kbrauer@mit.edu

conf = SparkConf().setAppName('project_spark')
sc = SparkContext(conf = conf)

# getting snapshot number from command line
snap = sys.argv[1]

# downsampling factor
down_max = 0.1

allpos_rdd = sc.emptyRDD()
snap3char = str(snap).zfill(3)
for i in range(64):
    # read the file into an numpy array
    newfile = h5py.File(
        '/mnt/s3/LX13/snapdir_'+snap3char+'/snap_'+snap3char+'.'+str(i)+'.hdf5','r')
    particletypes = newfile.keys()[1:]
    # loop through all particle types
    for newtype in particletypes:
        if newtype == 'PartType1':
            # load the coordinates of the high-resolution type into an rdd
            positions = newfile[newtype]['Coordinates'][:]
            positions_rdd = sc.parallelize(positions)
            # downsample the rdd
            typeindex = int(newtype[-1]) - 1
            subpositions_rdd = positions_rdd.sample(False,down_max)
            # concatenate them with the overall positions rdd
            allpos_rdd = allpos_rdd.union(subpositions_rdd)
        
finalarray = allpos_rdd.collect()
np.save("/mnt/s3/output"+str(snap), finalarray)