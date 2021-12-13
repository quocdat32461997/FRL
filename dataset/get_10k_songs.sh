# createe and cd into 10k_songs
mkdir 10_songs
cd 10k_songs

# download zip file
wget --no-parent https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
wget --no-parent http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip

# unzip to 10k_songs
unzip YearPredictionMSD.txt.zip
unzip train_triplets.txt.zip

