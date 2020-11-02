#!/bin/bash

# Use in colab to get the wikiart dataset.

# Install aria2 if it isn't already installed.
if ! dpkg -l aria2 &> /dev/null ; then
  sudo apt install aria2
fi 

# Use max connections for fastest download of the dataset.
aria2c -x 16 -s 16 http://ia802804.us.archive.org/14/items/wikiart-dataset/wikiart.tar.gz

# Extract and remove tarball. 
tar -xvf wikiart.tar.gz
rm wikiart.tar.gz