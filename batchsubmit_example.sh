#!/bin/bash

# This script is a simple example of how you can 
# batch submit a bunch of snapshots to be processed on Spark.

# Here, snapshots 0 to 20 are being processed. 
# This can be changed to process whichever snapshots you'd like.
for i in {0..20}
do
   spark-submit project_spark.py $i
done