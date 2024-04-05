from tqdm import tqdm
import torch


def train(model, device, train_loader, optimizer, loss_function, gradient_accumulation_steps, epochs, val_loader=None):
    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batch") as pbar:
            # Set training mode
            model.train()
            for batch_idx, (data, target) in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)

                # Compute loss and perform backpropagation
                output = model(data)
                train_loss = loss_function(output, target)
                train_loss.backward()

                # Optimize and reset accumulated gradients after gradient_accumulation_steps batches
                if batch_idx % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Evaluation
                if batch_idx == len(pbar) - 1 and val_loader is not None:
                    # Set evaluation mode
                    model.eval()
                    total = 0
                    running_loss = 0.0
                    with torch.no_grad():
                        for val_batch_idx, (data, target) in enumerate(val_loader):
                            data, target = data.to(device), target.to(device)

                            output = model(data)
                            val_loss = loss_function(output, target)
                            total += target.size(0)
                            running_loss = running_loss + val_loss.item()
                        mean_val_loss = running_loss / val_batch_idx

                    # Update progress bar with validation loss
                    pbar.set_postfix(loss=train_loss.item(), val_loss=mean_val_loss)

                else:
                    # Update progress bar with train loss
                    pbar.set_postfix(train_loss=train_loss.item())

    return model
