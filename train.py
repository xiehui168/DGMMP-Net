import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, epochs=100, device='cuda'):
    model.to(device)
    # Supp A.5: AdamW, lr=1e-4, weight_decay=1e-5
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = MultiTaskLoss()
    early_stopping = EarlyStopping(patience=20)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ct, dose, path, clin, times, events, rec, tox = [b.to(device) for b in batch]
            optimizer.zero_grad()
            # 训练时 mc_passes=1
            outputs = model(ct, dose, path, clin, mc_passes=1) 
            loss, _ = criterion(outputs, times, events, rec, tox)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_risks, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, dose, path, clin, times, events, rec, tox = [b.to(device) for b in batch]
                # 验证时计算均值，mc_passes=10 (Supp A.4)
                outputs = model(ct, dose, path, clin, mc_passes=10)
                val_risks.append(outputs['risk_mean'].cpu())
                val_times.append(times.cpu())
                val_events.append(events.cpu())
        
        val_risks = torch.cat(val_risks)
        val_times = torch.cat(val_times)
        val_events = torch.cat(val_events)
        
        c_index = calculate_c_index(val_risks, val_times, val_events)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val C-Index: {c_index:.4f}")
        
        early_stopping(c_index, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    # 加载最佳模型
    model.load_state_dict(early_stopping.best_model_wts)
    return model