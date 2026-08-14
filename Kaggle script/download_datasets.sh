export KAGGLE_USERNAME="tranquocbao2012"
export KAGGLE_KEY="KGAT_cfd0301101ae613eb63bbfe29a7ef502"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
mkdir -p ~/.kaggle

export PATH="$HOME/.local/bin:$PATH"

echo "Downloading synthetic-asl-numbers..."
mkdir -p synthetic-asl-numbers
kaggle datasets download -d lexset/synthetic-asl-numbers -p synthetic-asl-numbers --unzip

echo "Downloading wlasl-processed..."
mkdir -p wlasl-processed
kaggle datasets download -d risangbaskoro/wlasl-processed -p wlasl-processed --unzip

echo "Downloading chicagofswild..."
mkdir -p chicagofswild
kaggle datasets download -d joebeachcapital/chicagofswild -p chicagofswild --unzip

echo "Downloading asl-citizen..."
mkdir -p asl-citizen
kaggle datasets download -d abd0kamel/asl-citizen -p asl-citizen --unzip

echo "Downloading how2sign-holistic..."
mkdir -p how2sign-holistic
kaggle datasets download -d psewmuthu/how2sign-holistic -p how2sign-holistic --unzip

echo "All datasets downloaded successfully!"
