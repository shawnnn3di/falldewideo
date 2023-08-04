from utils import include_args
from dataparse import build_loader
from models import eventlevel, framelevel

import argparse
import pandas as pd
import pickle as pk
import random
import torch
import tqdm

def backward(optimizer, loss):
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run():
    
    '''
    Training and validating are done in this function. Mainly, three things are done in this function:
        1. parse the training settings, e.g., learning rate, data source, type of model, etc.,
        2. build a DataLoader and a model,
        3. train the model on the DataLoader and validate at a set frequency.
    '''
    
    '''
        1.
    '''
    parser = argparse.ArgumentParser()
    parser = include_args(parser)
    args = parser.parse_args()
    
    '''
        2.
        Note that if you want to modify the model structure, you can simply write a build_model function specifying  
            a. what is the model like,
            b. how to train your model,
        in a new file in model/, and import yours instead of the exsiting one. Similarly for a different dataset or 
        dataloader.
    '''
    if args.loadermode == 'framelevel':
        model, optimizer, scheduler, forward = framelevel.build_model(args.device, args.checkpoint, args.lr, 
                                                                      args.weight_decay)
    elif args.loadermode == 'eventlevel':
        model, optimizer, scheduler, forward = eventlevel.build_model(args.device, args.checkpoint, args.lr, 
                                                                      args.weight_decay)
    
    '''
        3.
        Train your model for num_epoch epochs. During training, for every valid_gap epochs, performance of the model is 
        validated. Files will be dumped, which can be visualized easily in evaluate.ipynb.
        
        In an epoch (train or epoch), a "batch" is taken out from the dataloader. the "forward" function digests the 
        "batch", put the digested "batch" into the model, and return the model's outputs. If it is a valid epoch, then 
        output will be dumped, if it is the right epoch to do so.
    '''
    randomidx = pd.DataFrame.from_dict({'idx': list(range(int(4020 / args.eventlength)))}).sample(frac=1.0)
    for epoch in range(args.num_epoch):
        
        ipack = random.randint(0, 9)
        
        print('pack %d in enumeration.' % ipack)
        
        train_loader, valid_loader = build_loader(args.pathcsi + '_%d.pk' % ipack, args.pathmask + '_%d.pk' % ipack, 
                                                  args.pathpose + '_%d.pk' % ipack, args.samplelength,
                                                  args.eventlength, args.testfrac, args.trainfrac, args.batch_size, 
                                                  args.num_workers, args.loadermode, randomidx['idx'])
        if epoch % args.valid_gap == 0:
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader))
                lenpbar = len(pbar)
                
                for idx, batch in pbar:
                    loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img = forward(batch, args.device, 
                                                                                                 model, False)
                    
                    pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f, %.4f. %.4f' % (
                        'valid', epoch, args.num_epoch, idx, lenpbar, loss_sm, loss_jhm, loss_paf))
                    torch.save(model, '%s/%s_%d.model' % (args.dump_path, args.comment, epoch))
                    
                    pk.dump([
                        jhm.detach().cpu().numpy(), 
                        y_jhm.detach().cpu().numpy(), 
                        paf.detach().cpu().numpy(), 
                        y_paf.detach().cpu().numpy(), 
                        sm.detach().cpu().numpy(), 
                        y_sm.detach().cpu().numpy(),
                        batch['name']], open('%s/valid_p%d_%s_%d_%d.pk' % (args.dump_path, ipack, args.comment, epoch, 
                                                                           idx), 'wb'))
        '''
            If it is a train epoch, then for every batch, SGD will be done. Results and model statedict will be dumped 
            if it is the right time to do so.
        '''
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        lenpbar = len(pbar)
        for idx, batch in pbar:
            loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img = forward(batch, args.device, model, 
                                                                                         False)
            loss = loss_sm + loss_jhm + loss_paf
            backward(optimizer, loss)
            
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f, %.4f. %.4f' % (
                'train', epoch, args.num_epoch, idx, lenpbar, loss_sm, loss_jhm, loss_paf))
            
            if epoch % args.valid_gap == 0:
                torch.save(model, '%s/%s_%d.model' % (args.dump_path, args.comment, epoch))
            
                if idx % args.traindumpgap == 0:
                    pk.dump([
                        jhm.detach().cpu().numpy(), 
                        y_jhm.detach().cpu().numpy(), 
                        paf.detach().cpu().numpy(), 
                        y_paf.detach().cpu().numpy(), 
                        sm.detach().cpu().numpy(), 
                        y_sm.detach().cpu().numpy(),
                        batch['name']], open('%s/train_p%d_%s_%d_%d.pk' % (args.dump_path, ipack, args.comment, epoch, 
                                                                           idx), 'wb'))

            
if __name__ == '__main__':
    run()