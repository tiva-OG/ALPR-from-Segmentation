import os
import torch
from os import path

def save(model, optimizer, epoch, loss, args):
    """ Saves model to checkpoint """
    
    state = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'loss': loss, 
        'epoch': epoch, 
    }
    
    checkpoint_dir = path.join(args.path, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, path.join(checkpoint_dir, args.name))
    
    # Save arguments
    summary_filename = path.join(checkpoint_dir, args.name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write('ARGUMENTS\n')
        
        for arg in sorted_args:
            arg_str = f'{arg}: {getattr(args, arg)}\n'
            summary_file.write(arg_str)

def load(model, optimizer, args):
    """ Loads model from checkpoint """
    
    device = torch.device(args.device)
    # path = path.join(args.path, 'checkpoint/EPOCH_10')
    state = torch.load(args.weights, map_location=device)

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    best_loss = state['loss']

    return model, optimizer, start_epoch, best_loss
