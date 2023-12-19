# To train a CSRNet Model

import torch.optim as optim

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()