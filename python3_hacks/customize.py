import tqdm

epochs = 10
steps_per_epoch = 100
data_loader = range(steps_per_epoch)

for epoch in range(epochs):
    with tqdm.tqdm(total=steps_per_epoch, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for batch in data_loader:
            print('Training...')
            pbar.update(1)