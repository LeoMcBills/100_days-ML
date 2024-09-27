import time
import tqdm

# Simulated data loader
def data_loader(num_batches):
    for _ in range(num_batches):
        # Simulating some processing time for each batch
        time.sleep(0.1)
        yield "batch_data"

# Simulated model training function for each batch
def train_batch(batch):
    # Simulating a training step
    time.sleep(0.05)
    return "trained_batch"

# Main training loop
def train_model(num_epochs, steps_per_epoch):
    for epoch in range(num_epochs):
        # Setting up the progress bar for each epoch
        with tqdm.tqdm(total=steps_per_epoch, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100) as pbar:
            for batch in data_loader(steps_per_epoch):
                # Train on each batch
                train_batch(batch)
                
                # Update the progress bar after each batch
                pbar.update(1)
                
        # Simulating the completion of an epoch (e.g., saving model checkpoints)
        print(f"Epoch {epoch+1} completed.\n")

# Parameters
num_epochs = 5      # Number of epochs
steps_per_epoch = 100  # Number of batches per epoch (data points)

# Start training
train_model(num_epochs, steps_per_epoch)
