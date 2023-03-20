import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from dataset import ASLData

from utils import *

def run_train(model, device, train_metric, train_loader, optimizer, scheduler, criterion, epoch, global_step, writer):
    model.train()
    train_loss = []
    train_bar = train_loader
    for x,y in train_bar:
        x = x.float().to(device)
        y = y.long().to(device)
        logits, arcface = model(x, y)
        arcface_loss = nn.CrossEntropyLoss()(arcface, y)
        loss = criterion(logits, y) * 0.5 + arcface_loss * 0.5
        train_metric.update(torch.argmax(logits, dim=1).detach().cpu(), y.detach().cpu())
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('lr', get_lr(optimizer), global_step)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss.append(loss.item())
        scheduler.step()
        global_step += 1
    train_loss = np.mean(train_loss)
    train_acc = train_metric.compute()
    train_metric.reset()
    writer.add_scalar('train/epoch_loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    log.info(f"Epoch:{epoch} > Train Loss: {train_loss:.04f}, Train Acc: {train_acc:0.04f}")
    return global_step

def run_val(model, device, val_metric, val_loader, criterion, epoch, writer):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for x,y in val_loader:
            x = x.float().to(device)
            y = y.long().to(device)
            logits, arcface = model(x, y)
            arcface_loss = nn.CrossEntropyLoss()(arcface, y)
            loss = criterion(logits, y) * 0.5 + arcface_loss * 0.5
            loss = criterion(logits, y)
            val_metric.update(torch.argmax(logits, dim=1).detach().cpu(), y.detach().cpu())
            val_loss.append(loss.item())
                            
    val_loss = np.mean(val_loss)
    val_acc = val_metric.compute()
    val_metric.reset()
    writer.add_scalar('val/epoch_loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    log.info(f"Epoch:{epoch} > Val Loss: {val_loss:.04f}, Val Acc: {val_acc:0.04f}")
    return val_loss, val_acc

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="v2")
def main(config: DictConfig) -> None:
    SEED = config.seed
    set_seed(SEED)
    
    datax = np.load(config.dataset.feature_data)
    datay = np.load(config.dataset.feature_labels)
    df = pd.read_csv('data/train.csv')
    trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.2, random_state=SEED, stratify=df['participant_id'].values)

    train_data = ASLData(trainx, trainy)
    valid_data = ASLData(testx, testy)

    EPOCHS = config.training.epoch
    train_loader = DataLoader(
        train_data,
        batch_size=config.dataset.train_batch_size, 
        num_workers=config.dataset.num_workers, 
        shuffle=True
    )
    val_loader = DataLoader(
        valid_data,
        batch_size=config.dataset.val_batch_size,
        num_workers=config.dataset.num_workers, 
        shuffle=False
    )
    
    device = config.device
    loss_fn = instantiate(config.model.loss)
    model = instantiate(config.model.arch, loss_fn=loss_fn, arcface=config.model.arcface).to(device)
    optimizer = instantiate(config.training.optimizer, params=model.parameters())
    criterion = instantiate(config.training.criterion)
    scheduler = instantiate(config.training.scheduler, optimizer=optimizer)
    best_metric = 0.0
    
    output_dir = os.path.join(HydraConfig.get().run.dir)
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(str(output_dir))
    
    train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=config.dataset.out_features)
    val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=config.dataset.out_features)
    
    global_step = 0
    for epoch in range(EPOCHS):
        global_step = run_train(model, device, train_metric, train_loader, optimizer, scheduler, criterion, epoch, global_step, writer)
        val_loss, val_acc = run_val(model, device, val_metric, val_loader, criterion, epoch, writer)
        if val_acc > best_metric:
            log.info(f"SAVING CHECKPOINT: val_metric {best_metric:0.04f} -> {val_acc:0.04f}")
            best_metric = val_acc
            checkpoint = create_checkpoint(model, optimizer, epoch, scheduler=scheduler)
            torch.save(
                checkpoint,
                f"{output_dir}/best.pth",
            )
        log.info("="*50)
    writer.close()
    
if __name__ == '__main__':
    main()