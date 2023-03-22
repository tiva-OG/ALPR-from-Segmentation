import torch
from tqdm import tqdm

class Train:

    def __init__(self, model, data_loader, optimizer, criterion, device):
        
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.dice, self.bce = criterion
        self.device = device
    
    def run_epoch(self, verbose=False):
        self.model.train()
        
        epoch_loss, epoch_dice, epoch_bce = (0, 0, 0)
        
        for step, batch in enumerate(self.data_loader, start=1):
            inputs = batch['image'].to(self.device)
            labels = batch['mask'].to(self.device)
            
            outputs = self.model(inputs)
            
            dice_loss = self.dice(outputs, labels)
            bce_loss = self.bce(outputs, labels)
            loss = dice_loss + bce_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += dice_loss.item()
            epoch_bce += bce_loss.item()
            
            if verbose:
                tqdm.write(f'[Step: {step:d}] Iteration loss: {loss.item():.4f}\n')
                # print(f'[Step: {step:d}] Iteration loss: {loss.item():.4f}\n')
        
        return (
            epoch_loss / len(self.data_loader), 
            epoch_dice / len(self.data_loader), 
            epoch_bce / len(self.data_loader)
            )

class Test:
    
    def __init__(self, model, data_loader, criterion, device):
        
        self.model = model
        self.data_loader = data_loader
        self.dice, self.bce = criterion
        self.device = device
    
    def run_epoch(self, verbose=False):
        self.model.eval()
        
        epoch_loss, epoch_dice, epoch_bce = (0, 0, 0)
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.data_loader)):
                inputs = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
                outputs = self.model(inputs)
                
                dice_loss = self.dice(outputs, labels)
                bce_loss = self.bce(outputs, labels)
                loss = dice_loss + bce_loss
                
                epoch_loss += loss.item()
                epoch_dice += dice_loss.item()
                epoch_bce += bce_loss.item()
                
                if verbose:
                    tqdm.write(f'[Step: {step:d}] Iteration loss: {loss.item():.4f}')
                    # print(f'[Step: {step:d}] Iteration loss: {loss.item():.4f}')
        
        return (
            epoch_loss / len(self.data_loader), 
            epoch_dice / len(self.data_loader), 
            epoch_bce / len(self.data_loader)
            )
