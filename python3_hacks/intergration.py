import pandas as pd
from tqdm import tqdm

tqdm.pandas()    # Adds progress bars to pandas operations

df = pd.DataFrame({'a': range(100000)})
df['a'].progress_apply(lambda x: x * 2)