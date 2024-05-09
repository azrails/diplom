#!/bin/bash/
fileid="14QcsxQILplN8nXSa87ACOOYiCebiVPxM"
filename="lmdb.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}