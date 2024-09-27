import tqdm
import time

outer = tqdm.tqdm(total=100, desc="Outer loop")
inner = tqdm.tqdm(total=100, desc="Inner loop", position=1)

for _ in range(100):
    for _ in range(100):
        time.sleep(0.01)
        inner.update(1)
    outer.update(1)
    inner.reset()