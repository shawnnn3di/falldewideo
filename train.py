from utils import include_args
from dataparse import build_loader
from models.framewise import build_model

import argparse
import pickle as pk
import torch
import tqdm

def backward(optimizer, loss):
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run():
    
    parser = argparse.ArgumentParser()
    parser = include_args(parser)
    args = parser.parse_args()
    
    train_loader, valid_loader = build_loader(args.pathcsi, args.pathmask, args.pathpose, args.eventlength, 
                                              args.testfrac, args.trainfrac, args.batch_size, args.num_workers, 
                                              args.loadermode)
    model, optimizer, scheduler, forward = build_model(args.device, args.checkpoint, args.lr, args.weight_decay)
    
    for epoch in range(args.num_epoch):
        if epoch % args.valid_gap == 0:
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader))
                lenpbar = len(pbar)
                
                for idx, batch in pbar:
                    loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img = forward(batch, args.device, model, False)
                    
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
                        batch['name']], open('%s/valid_%s_%d_%d.pk' % (args.dump_path, args.comment, epoch, idx), 'wb'))
                    
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        lenpbar = len(pbar)
        
        for idx, batch in pbar:
            loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img = forward(batch, args.device, model, False)
            loss = loss_sm + loss_jhm + loss_paf
            backward(optimizer, loss)
            
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f, %.4f. %.4f' % (
                'train', epoch, args.num_epoch, idx, lenpbar, loss_sm, loss_jhm, loss_paf))
            
            if epoch % args.valid_gap:
                torch.save(model, '%s/%s_%d.model' % (args.dump_path, args.comment, epoch))
            
                if idx % args.traindumpgap:
                    pk.dump([
                        jhm.detach().cpu().numpy(), 
                        y_jhm.detach().cpu().numpy(), 
                        paf.detach().cpu().numpy(), 
                        y_paf.detach().cpu().numpy(), 
                        sm.detach().cpu().numpy(), 
                        y_sm.detach().cpu().numpy(),
                        batch['name']], open('%s/train_%s_%d_%d.pk' % (args.dumpdir, args.comment, epoch, idx), 'wb'))
            
            
if __name__ == '__main__':
    run()