#!/bin/bash
# Use for colab to download and organize smaller art datasets from kaggle.

# First, check if Kaggle account is configured.
if ! kaggle -v &> /dev/null; then
  echo "Kaggle account not configured. Exiting script."
  exit
fi

# Function for downloading simply-structured datasets.
get_dataset () {
  ZIP_NAME="${1##*/}.zip"
  kaggle datasets download -d ${1}
  unzip ${ZIP_NAME} -d datasets
  rm ${ZIP_NAME}
}

# Function for combining train/test directories without duplicate names.
style_combiner () {
  for dir in ${1}/*
  do
    COUNTER=0
    STYLE="${dir##*/}"
    for file in ${dir}/*
    do
      mv "${file}" "${2}/${STYLE}/${COUNTER}.jpg"
      let COUNTER+=1
    done
  done
}

# Create top-level directory
mkdir datasets

# Download and organize all simply-structured datasets.
SIMPLES=("greg115/abstract-art" "bryanb/abstract-art-gallery" "olgabelitskaya/art-pictogram" "rickyjli/chinese-fine-art")
for ds in ${SIMPLES[@]}
do
  get_dataset ${ds}
done

# Cleanup for Chinese Fine Art dataset.
mv datasets/Dataset datasets/chinese_fine_art
rm datasets/artists.csv
rm datasets/artworks.csv

# Download and organize "art_images" dataset
kaggle datasets download -d thedownhill/art-images-drawings-painting-sculpture-engraving
unzip art-images-drawings-painting-sculpture-engraving.zip dataset/dataset_updated/* -d datasets
rm art-images-drawings-painting-sculpture-engraving.zip
style_combiner datasets/dataset/dataset_updated/validation_set datasets/dataset/dataset_updated/training_set 
mv datasets/dataset/dataset_updated/training_set datasets/art_images
rm -r datasets/dataset

# Download and organize "Best Artworks of All Time" dataset
kaggle datasets download -d ikarus777/best-artworks-of-all-time
unzip best-artworks-of-all-time.zip images/\* -d datasets
rm best-artworks-of-all-time.zip
mv datasets/images/images datasets/baat
rm -r datasets/images