set -e

echo "Which dataset do you want to download?"
echo "  1) SampleSet (6GB) "
echo "  2) Rectified (123GB) "
echo "  3) Cleaned   (136GB) "
echo ""

read -p "Enter 1, 2, or 3: " choice

URL=""
FILENAME=""

if [ "$choice" == "1" ]; then
    URL="http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
    FILENAME="SampleSet.zip"
elif [ "$choice" == "2" ]; then
    URL="http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip"
    FILENAME="Rectified.zip"
elif [ "$choice" == "3" ]; then
    URL="http://roboimagedata2.compute.dtu.dk/data/MVS/Cleaned.zip"
    FILENAME="Cleaned.zip"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

cd "$(dirname "$0")/.."

echo "Creating data/dtu directory..."
mkdir -p data/dtu
cd data/dtu

echo "Downloading $FILENAME..."

curl -L -C - "$URL" -o "$FILENAME"

echo " Extracting $FILENAME..."
unzip -q "$FILENAME"

echo "Cleaning up zip file to save space..."

rm "$FILENAME"

echo "Done! Data is ready inside the 'data/dtu' folder."