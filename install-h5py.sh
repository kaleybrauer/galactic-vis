#!/bin/bash

sudo /usr/bin/yum -y install python-devel
sudo /usr/bin/yum -y install –y epel-release
sudo /usr/bin/yum-config-manager --enable epel
sudo /usr/bin/yum -y install hdf5-devel
sudo easy_install h5py