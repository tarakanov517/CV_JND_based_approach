import torch
import numpy as np

@torch.no_grad()
def collect_simkin(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for imgs, y in loader:
        imgs = tuple(t.to(device) for t in imgs)
        preds.append(model(imgs, mode='simk').squeeze(1).cpu())
        tgts.append(y)
    return torch.cat(preds).numpy(), torch.cat(tgts).numpy()

def choose_thres(preds, psi):
    true_vis = psi > 0
    P, N = true_vis.sum(), (~true_vis).sum()
    for t in np.linspace(-1.5, 1.5, 31):
        dec = preds > t
        tpr = (dec & true_vis).sum() / P
        tnr = (~dec & ~true_vis).sum() / N
        bal = 0.5 * (tpr + tnr)
        print(f"t={t:+.2f}  bal_acc={bal:.3f}  TPR={tpr:.2f}  TNR={tnr:.2f}")