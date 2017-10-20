import os
import glob


for csv in glob.glob(r'*.csv'):
    with open(csv, 'r') as src:
        with open(csv[:-4] + '.mat', 'w') as dst:
            dst.write(src.read().replace(',', ' ').replace('?', 'NaN'))
