import argparse


def include_args(parser: argparse.ArgumentParser):
    parser.add_argument('--shuffle_train',  default=True, type=bool)
    parser.add_argument('--batchsize',      default=32, type=int)
    parser.add_argument('--lr',             default=1e-3, type=float)
    parser.add_argument('--weight_decay',   default=1e-5, type=float)
    parser.add_argument('--lr_gamma',       default=0.99, type=float)
    parser.add_argument('--num_epoch',      default=300, type=int)
    parser.add_argument('--valid_gap',      default=1, type=int)
    parser.add_argument('--preview_gap',    default=100, type=int)
    parser.add_argument('--checkpoint_gap', default=20, type=int)
    parser.add_argument('--dump_loss_gap',  default=-1, type=int)
    
    parser.add_argument('--half',           default=False, type=bool)
    parser.add_argument('--num_workers',    default=12, type=int)
    parser.add_argument('--gpuid',          default=0, type=int)
    
    parser.add_argument('--comment',        default='test', type=str)
    
    parser.add_argument('--prefix',         default='annotate/annotate', type=str)
    
    parser.add_argument('--checkpoint',     default='checkpoint', type=str)
    
    return parser