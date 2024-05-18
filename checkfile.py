import pandas as pd
import numpy as np
import lasio

#load las file
las = lasio.read('/Users/rianrachmanto/pypro/data/K-4.las')
#convert to dataframe
df = las.df()
print(df.head())