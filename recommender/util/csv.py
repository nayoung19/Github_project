import pandas as pd

data = pd.read_csv('../files/info.csv')
data = data.drop_duplicates(keep='first', inplace='False')
data.to_csv('userinfo-new')