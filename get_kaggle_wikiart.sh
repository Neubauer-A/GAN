#!/bin/bash
# Use for colab to get kaggle version of wikiart dataset.

# First, check if Kaggle account is configured.
if ! kaggle -v &> /dev/null; then
  echo "Kaggle account not configured. Exiting script."
  exit
fi

mkdir wikiart
kaggle datasets download -d ipythonx/wikiart-gangogh-creating-art-gan
unzip wikiart-gangogh-creating-art-gan.zip -d wikiart
rm wikiart-gangogh-creating-art-gan.zip