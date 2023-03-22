import torch
import warnings
import torch.nn as nn

from os import path
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from model import UNet
from args import train_args
from metrics import DiceLoss
from data import LicensePlates
from src import Train, Test, checkpoint, display
from data.loader import load_files, split_dataset


warnings.filterwarnings('ignore')

args = train_args()
DEVICE = torch.device(args.device)
DATA_DIR = path.join(args.path, 'dataset')
LOADER_ARGS = dict(batch_size=args.batch_size, num_workers=args.workers)


def load_dataset():
    print('\nLoading Dataset...')
    
    if args.mode == 'train':
        train_ids, val_ids = split_dataset(DATA_DIR, args.split)
        
        train_set = LicensePlates(DATA_DIR, train_ids, 'train')
        train_loader = DataLoader(train_set, shuffle=True, **LOADER_ARGS)
        
        val_set = LicensePlates(DATA_DIR, val_ids, 'val')
        val_loader = DataLoader(val_set, shuffle=False, **LOADER_ARGS)
        
        print(f'\nTrain size: {len(train_set)}')
        print(f'Validation size: {len(val_set)}\n')
        
        iterator = iter(train_loader)
        sample = next(iterator)
        images, masks = sample['image'], sample['mask']
        
    else:
        test_ids = load_files(DATA_DIR)
        
        test_set = LicensePlates(DATA_DIR, test_ids, 'test', args.size)
        test_loader = DataLoader(test_set, shuffle=False, **LOADER_ARGS)
        
        print(f'\nTest size: {len(test_set)}')
        
        iterator = iter(test_loader)
        sample = next(iterator)
        images, masks = sample['image'], sample['mask']
    
    if args.show_batch:
        display.show_batch(images, masks)
    
    data_loader = (train_loader, val_loader) if args.mode == 'train' else test_loader
    
    return data_loader

def train(data_loader):
    print(f'\r\n***** TRAINING *****')
    
    train_loader, val_loader = data_loader
    model = UNet(args.classes)
    
    dice_loss = DiceLoss(from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    criterion = (dice_loss, bce_loss)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    
    if args.weights:
        model, optimizer, start_epoch, best_loss = checkpoint.load(model, optimizer, args)
        print(f'\r\nResuming training from epoch {start_epoch} | Best loss was {best_loss:.4f}\n')
    else:
        start_epoch = 1
        best_loss = 1000
    
    print()
    _train = Train(model, train_loader, optimizer, criterion, DEVICE)
    _validate = Test(model, val_loader, criterion, DEVICE)
    
    for epoch in range(start_epoch, args.epochs+2):
        print(f'\n***** EPOCH: {epoch} *****\n')
        
        train_loss, train_dice, train_bce = _train.run_epoch(args.verbose)
        scheduler.step(train_loss)
        print(f'\n< Dice Loss: {train_dice:.4f}; BCE Loss: {train_bce} >\n')
        
        if (epoch%5 == 0) or (epoch == args.epochs+2):
            print(f'\n===== VALIDATING =====')
            val_loss, val_dice, val_bce = _validate.run_epoch(args.verbose)
            print(f'\n< Dice Loss: {val_dice:.4f}; BCE Loss: {val_bce} >\r')
            
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint.save(model, optimizer, epoch, best_loss, args)
    
    checkpoint.save(model, optimizer, epoch, best_loss, args)

def test(model, test_loader):
    print(f'\r\n***** TESTING *****')
    
    dice_loss = DiceLoss(from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    criterion = (dice_loss, bce_loss)
    
    _test = Test(model, test_loader, criterion, DEVICE)
    _, dice, bce = _test.run_epoch(args.verbose)
    print(f'\n< Dice Loss: {dice:.4f}; BCE Loss: {bce} >')
    
    if args.show_batch:
        pass

if __name__ == '__main__':
    assert path.isdir(DATA_DIR), f'The directory {DATA_DIR} does not exist'
    
    data_loader = load_dataset()

    if args.mode == 'train':
        train(data_loader)
    else:
        model = UNet(args.classes)
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        model, optimizer, start_epoch, best_loss = checkpoint.load(model, optimizer, args)
        test(model, data_loader)
