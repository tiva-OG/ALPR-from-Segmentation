from argparse import ArgumentParser


def train_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        '--mode', '-m', 
        choices=['train', 'test'], 
        default='train', 
        help='`train`: Performs training and validation;\n`test`: Tests the pre-trained model.'
        )
    
    parser.add_argument(
        '--path', '-p', 
        type=str, 
        default='.', 
        help='Home directory containing all necessary files and folders.'
        )
    
    parser.add_argument(
        '--epochs', '-e', 
        type=int, 
        default=300, 
        help='Number of epochs to train for.'
        )
    
    parser.add_argument(
        '--batch_size', '-bs', 
        type=int, 
        default=1, 
        help='Batch size.'
        )
    
    parser.add_argument(
        '--learning_rate', '-lr', 
        type=float, 
        default=5e-4, 
        help='Learning rate.'
        )
    
    parser.add_argument(
        '--weight_decay', '-wd',
        type=float, 
        default=2e-4, 
        help='L2 regularization factor.'
        )
    
    parser.add_argument(
        '--device', 
        choices=['cpu', 'cuda'], 
        default='cpu', 
        help='Device to train network on.'
        )
    
    parser.add_argument(
        '--split',
        type=float, 
        default=0.25, 
        help='Ratio of data to use for validation.'
        )
    
    parser.add_argument(
        '--workers', 
        type=int, 
        default=4, 
        help='Number of sub-processes to use in loading data.'
        )
    
    parser.add_argument(
        '--verbose', '-v', 
        type=bool,
        default=True, 
        help='Print loss for every step'
        )
    
    parser.add_argument(
        '--show_batch', 
        action='store_true', 
        help='Display image samples when loading the dataset or making predictions.'
        )
    
    parser.add_argument(
        '--classes', 
        type=int, 
        default=1, 
        help='Number of classes.'
        )
    
    parser.add_argument(
        '--size', 
        type=tuple, 
        default=(572, 572), 
        help='Resize image.'
        )
    
    parser.add_argument(
        '--weights',  
        type=str, 
        default='data/unet/final_checkpoint', 
        help='Path to pre-trained weights.'
        )
    
    return parser.parse_args()

