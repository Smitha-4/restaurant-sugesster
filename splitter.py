
import pandas as pd

data = pd.read_csv("clean_file.csv")
# no of csv files with row size
k = 6
size = 10000
 
for i in range(k):
    df = data[size*i:size*(i+1)]
    df.to_csv(f'clean_file_{i+1}.csv', index=False)
print('done')