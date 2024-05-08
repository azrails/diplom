import os
import json
import torch
from utils import config
from model.vit import StageOneEncoder

def save_checkpoint(path_to_checkpoints_folder, checkpoint_name, conf, model, optimizer, scheduler, train_losses, val_scores):
    """
    Save model, config, score
    """
    path_to_checkpoint = os.path.join(path_to_checkpoints_folder, checkpoint_name)
    os.makedirs(path_to_checkpoint, exist_ok=True)
    config.dump_config(path_to_checkpoint, conf, 'config.yaml')

    path_to_metrics = os.path.join(path_to_checkpoint, 'metrics.json')
    with open(path_to_metrics, 'w') as f:
        json.dump({'train_losses': train_losses, 'val_scores': val_scores}, f, indent=4)
    
    path_to_model = os.path.join(path_to_checkpoint, 'model.pt')
    torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                
                },
                path_to_model
                )


def load_checkpoint(path_to_checkpoints_folder, checkpoint_name, device):
    path_to_data = os.path.join(path_to_checkpoints_folder, checkpoint_name)
    conf = config.load_config(os.path.join(path_to_data, 'config.yaml'))

    with open(os.path.join(path_to_data, 'metrics.json'), 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    val_scores = data['val_scores']

    path_to_model = os.path.join(path_to_data, 'model.pt')
    model_state = torch.load(path_to_model)
    model = StageOneEncoder(**conf['model']['VITEncoder'])
    model.load_state_dict(model_state['model_state_dict'])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), **conf['optimizer_params'])
    optimizer.load_state_dict(model_state['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=conf['train_settings']['milestones'], 
        gamma=conf['train_settings']['lr_decay'])
    return model, optimizer, scheduler, conf, train_losses, val_scores