import tqdm
import time

with tqdm.tqdm(total=100) as pbar:
    for i in range(10):
        time.sleep(1)
        pbar.update(10)