# Baseline: FedAvg with IID MNIST Dataset

Implementation target: a Google Colab notebook optimized for fast execution on a T4 GPU.

## Configuration

- Framework: PyTorch.
- Dataset: MNIST.
- Number of clients: 25.
- Training rounds: 40.
- Local epochs per round: 3.
- Local batch size: 64.
- Local optimizer: SGD.
- Learning rate: 0.01.
- Server aggregation: standard FedAvg using client dataset sizes as weights.
- Participation: all 25 clients participate in every round.
- No runtime device-selection logic is required; the notebook assumes a Colab T4 GPU.
- Use `FLasIRL_new` as the Drive project folder name to avoid shell issues with parentheses.
- Use separate seeds for partitioning, model initialization, and batch ordering; save the seed bases in run metadata.
- Never use one shared global RNG for batch ordering. Derive a deterministic batch seed from `(batch_seed, round, client_id, local_epoch)` and create a fresh local `torch.Generator` for that combination.

## Step 1: Imports and setup

- Import PyTorch, torchvision, NumPy, Matplotlib, and utilities for serialization and run metadata.
- Set random seeds for reproducibility.
- Mount Google Drive.
- Define the project root:
  `MyDrive/FLasIRL_new`.
- Define separate folders for cached data, partitions, embeddings, charts, and baseline experiment runs (`baseline/run{run_id}`).

## Step 2: Download and load MNIST

- Use `MyDrive/FLasIRL_new/data/mnist` as the dataset directory.
- Download MNIST only when it is not already present.
- Use the standard MNIST train/test split.
- Use model-specific preprocessing; do not share one normalization pipeline between the two model options.
- Keep the original MNIST tensor available so preprocessing can be performed efficiently on the GPU.

## Step 3: Create and cache an IID partition

- Split the training set IID across 25 clients.
- Store each client’s sample indices in a reusable partition file.
- If the partition file already exists, load it instead of recreating it.
- Validate that every training sample belongs to exactly one client.
- Within each local epoch, generate the client’s permutation from its derived local generator and split that permutation into batches. Do not depend on client execution order, global RNG state, or DataLoader worker RNG state.
- Keep the final incomplete batch (`drop_last=False`): each client has 2,400 samples, resulting in 37 batches of 64 and one batch of 32. This ensures every sample contributes to training.
- Draw a stacked bar chart showing each client’s class distribution and sample count.
- Save the chart to Google Drive.

## Step 4: Select and prepare the model

The notebook must allow the model option to be selected near the beginning.

### Case 1: Frozen ImageNet ResNet-18 embedding + linear classifier

- Load ImageNet-pretrained ResNet-18.
- Resize images to `112x112` by default, with `224x224` available as a configuration option. The resize is required because processing `28x28` images through ResNet-18 produces overly small feature maps and weak embeddings.
- Repeat the grayscale channel three times; do not change the pretrained first convolution.
- Normalize using ImageNet mean and standard deviation.
- Remove the original ResNet classification head and use the remaining network to produce one embedding vector per image.
- Freeze all ResNet parameters.
- During embedding extraction, call `backbone.eval()` and use `torch.no_grad()` so BatchNorm running statistics are not updated.
- Compute embeddings once for the MNIST train and test sets, then cache them in the dataset folder for reuse.
- Include the backbone name, pretrained-weight version, resize size, normalization type, and dataset split in the embedding-cache key.
- Train only a linear layer mapping the embedding vector to 10 MNIST classes.
- During federated training, each client trains its local copy of the linear layer on its cached embeddings. Only the linear-layer parameters are sent to the server and aggregated.

### Case 2: End-to-end CNN

- Use the following fixed CNN architecture:
  - `Conv2d(1, 32, kernel_size=3, padding=1)`
  - `ReLU`
  - `MaxPool2d(2)`
  - `Conv2d(32, 64, kernel_size=3, padding=1)`
  - `ReLU`
  - `MaxPool2d(2)`
  - `Flatten`
  - `Linear(64 * 7 * 7, 128)`
  - `ReLU`
  - `Linear(128, 10)`
- Do not use BatchNorm or Dropout in this baseline, avoiding extra state buffers and additional random operations during local training.
- Keep images at `28x28`, one channel, and normalize using MNIST mean and standard deviation.
- Train all CNN parameters locally on each client.
- Send the complete local model parameters to the server for FedAvg.

## Step 5: Run FedAvg

For each of the 40 rounds:

1. Send the current global model to all 25 clients.
2. Each client trains locally for 3 epochs using SGD, batch size 64, and learning rate 0.01.
3. Return each client’s updated parameters and number of training samples.
4. Aggregate parameters on the server with weighted FedAvg.
5. Evaluate the aggregated global model on the complete MNIST training set and test set.
6. Define `global_train_loss` as the loss of the aggregated global model on the complete training set. Save `global_train_loss`, `test_loss`, and `test_accuracy`.
7. Save a checkpoint after every round, including the global model state, completed round, optimizer state if present, and random-number-generator states.

For fast execution on the T4, keep MNIST tensors and cached embeddings on the GPU when memory permits. Use tensor indices to create client batches instead of PIL-based `DataLoader` preprocessing. Perform ResNet resize, channel repetition, and ImageNet normalization on the GPU in batches during the one-time embedding extraction. Reuse cached embeddings during all subsequent rounds.

All clients must start each round from the same global model state. Batch ordering must be generated from the deterministic `(round, client_id, local_epoch)` seed, so resuming from a checkpoint or changing client execution order produces the same batches. Since the baseline CNN and linear classifier contain no BatchNorm or other persistent buffers, FedAvg only needs to aggregate `model.parameters()`.

Append metrics to CSV or JSONL immediately after every round and flush the file. Since a checkpoint is saved after every round, a run interrupted at round 37 can resume directly from round 37 without repeating completed rounds. If an earlier checkpoint is selected manually, truncate metrics and checkpoints after that round before resuming.

## Step 6: Save experiment results

Create a unique run ID and save:

- Model option and architecture details.
- Number of clients, rounds, local epochs, batch size, learning rate, and optimizer.
- Partition seed, model seed, batch-order seed, and dataset/partition identifiers.
- Per-round `global_train_loss`, test loss, and test accuracy.
- Per-round checkpoints and final model parameters.
- Final accuracy as the primary reported accuracy.
- Best test accuracy only as a secondary diagnostic; do not use it as the main reported result.

Store all run outputs under `baseline/run{run_id}` in Google Drive (including metrics, checkpoints, final parameters, config, and summary) so that a run can be reproduced and identified later.

## Step 7: Plot results

- Add a notebook cell that loads a selected run by run ID.
- Plot global test accuracy versus communication round.
- Optionally plot training/test loss versus round.
- Save the charts to the run’s directory (`baseline/run{run_id}/metrics.png`).

## Expected notebook structure

1. Configuration and imports.
2. Google Drive mount and directory setup.
3. MNIST loading.
4. IID partition creation/loading and visualization.
5. Model selection and preparation.
6. Optional ResNet embedding creation/loading.
7. FedAvg training and checkpointing.
8. Result saving.
9. Result loading and visualization.
