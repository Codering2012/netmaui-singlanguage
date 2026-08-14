export KAGGLE_USERNAME="tranquocbao2012"
export KAGGLE_KEY="KGAT_cfd0301101ae613eb63bbfe29a7ef502"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
export PATH="$HOME/.local/bin:$PATH"

cd /dev/shm/datasets

echo "Downloading asl-citizen..."
mkdir -p asl-citizen
kaggle datasets download -d abd0kamel/asl-citizen -p asl-citizen --unzip

echo "Downloading how2sign-holistic..."
mkdir -p how2sign-holistic
kaggle datasets download -d psewmuthu/how2sign-holistic -p how2sign-holistic --unzip

echo "All remaining datasets downloaded successfully!"
