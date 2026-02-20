# model_evaluator.py
import torch
from torch import nn
from sklearn.metrics import accuracy_score

def eval_model(model: nn.Module, data_loader, lossFn: nn.Module, device='cpu'):

    model.eval()
    loss_avg, acc_avg = 0, 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_logit = model(X)
            loss = lossFn(y_logit, y)

            preds = torch.argmax(y_logit, dim=1)

            acc = accuracy_score(
                y.cpu().numpy(),
                preds.cpu().numpy()
            )

            acc_avg += acc
            loss_avg += loss.item()

        loss_avg /= len(data_loader)
        acc_avg /= len(data_loader)

    return {
        "Model name": model.__class__.__name__,
        "Model loss": loss_avg,
        "Model accuracy": acc_avg
    }
    
