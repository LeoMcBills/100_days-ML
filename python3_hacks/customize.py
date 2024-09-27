import tqdm
import time

epochs = 100
steps_per_epoch = 10
data_loader = range(steps_per_epoch)

for epoch in range(epochs):
    with tqdm.tqdm(total=steps_per_epoch, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for batch in data_loader:
            print('Training...')
            pbar.update(1)
        time.sleep(0.5)