#!/bin/bash
apt-get install p7zip
fileid="1uETI9hoZxbGchmB7D_o6WJaEfZLwx1nc"
filename="dataset.7z"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
echo "Downloaded yelp2013 dataset"
7zr x dataset.7z -y
echo "Unziped dataset.7z"
mv dataset/yelp-2013-seg-20-20.dev.ss ./
mv dataset/yelp-2013-seg-20-20.train.ss ./
mv dataset/yelp-2013-seg-20-20.test.ss ./
rm -r dataset/
rm cookie
rm dataset.7z
echo "Preprocess dataset"
python process_data.py
echo "Completed"
