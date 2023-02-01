# Get url from DVC
import dvc.api
from io import StringIO
import pandas as pd

path = 'data/wine-quality.csv'
repo = 'https://github.com/coucjh/wine'
version = 'v3'
# remote = 'myremote'

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)
data_read = dvc.api.read(
    path=path,
    repo=repo,
    rev=version
)
data_read = StringIO(data_read)
print(data_url)

data = pd.read_csv(data_read, sep=",", header=0)
data.to_csv("data/data_raw.csv")