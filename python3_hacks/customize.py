import tqdm
import time

for i in tqdm.tqdm(range(5), desc="outer Loop"):
    for j in tqdm.tqdm(range(100), desc="inner loop", leave=False):
        time.sleep(0.01)