#!/bin/bash
export KAGGLE_USERNAME="tranquocbao2012"
export KAGGLE_KEY="KGAT_cfd0301101ae613eb63bbfe29a7ef502"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p /mnt/ramdisk/datasets
cd /mnt/ramdisk/datasets

echo "[1/6] Downloading ASL Alphabet..."
mkdir -p asl-alphabet
kaggle datasets download -d grassknoted/asl-alphabet -p asl-alphabet
unzip -q -o asl-alphabet/*.zip -d asl-alphabet/
rm asl-alphabet/*.zip

echo "[2/6] Downloading Synthetic ASL Numbers..."
mkdir -p synthetic-asl-numbers
kaggle datasets download -d lexset/synthetic-asl-numbers -p synthetic-asl-numbers
unzip -q -o synthetic-asl-numbers/*.zip -d synthetic-asl-numbers/
rm synthetic-asl-numbers/*.zip

echo "[3/6] Downloading WLASL..."
mkdir -p wlasl-processed
kaggle datasets download -d risangbaskoro/wlasl-processed -p wlasl-processed
unzip -q -o wlasl-processed/*.zip -d wlasl-processed/
rm wlasl-processed/*.zip

echo "[4/6] Downloading ChicagoFSWild..."
mkdir -p chicagofswild
kaggle datasets download -d joebeachcapital/chicagofswild -p chicagofswild
unzip -q -o chicagofswild/*.zip -d chicagofswild/
rm chicagofswild/*.zip
echo "Extracting ChicagoFSWild frames..."
cd chicagofswild
tar -xzf ChicagoFSWild-Frames.tgz
rm ChicagoFSWild-Frames.tgz
cd ..

echo "[5/6] Downloading ASL Citizen..."
mkdir -p asl-citizen
kaggle datasets download -d abd0kamel/asl-citizen -p asl-citizen
unzip -q -o asl-citizen/*.zip -d asl-citizen/
rm asl-citizen/*.zip

echo "[6/6] Downloading How2Sign Holistic..."
mkdir -p how2sign-holistic
kaggle datasets download -d psewmuthu/how2sign-holistic -p how2sign-holistic
unzip -q -o how2sign-holistic/*.zip -d how2sign-holistic/
rm how2sign-holistic/*.zip

echo "ALL DATASETS DOWNLOADED AND EXTRACTED SUCCESSFULLY!"
