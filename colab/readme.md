# Google Colab Stuff

This folder contains some example scripts for training an LSTM using Google Colab.  

To successfully run the Colab notebook:

Make a directroy in the `My Drive` folder of your Google Drive named `backups`. This folder will stored processed training data and LSTM model checkpoints.

Clone https://github.com/louisabraham/python3-midi/tree/louisabraham-patch-1 repo to `My Drive`.

Copy the `MuGen` folder to `My Drive`.

Copy `train_colab.py` to `My Drive`.

Copy some midi sequences for training to a folder on your Google Drive `My Drive/data`

Open up `colab_notebook.ipynb` in google colab and execute the cells.

The last cell will execute `train_colab.py`.  
It is possible to add the filename of a model in `My Drive/backups` as an argument when running `train_colab.py`. This is for continuing/resuming from a checkpoint.

** lines 105 to 135 in `train_colab.py` will likely need altering in order to point at your midi data depending on how you organize the file structure in `My Drive/data`.