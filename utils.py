import argparse


def include_args(parser: argparse.ArgumentParser):
    parser.add_argument('--shuffle_train',      default=True, type=bool)
    parser.add_argument('--batch_size',         default=32, type=int)
    parser.add_argument('--lr',                 default=1e-3, type=float)
    parser.add_argument('--weight_decay',       default=1e-5, type=float)
    parser.add_argument('--lr_gamma',           default=0.99, type=float)
    parser.add_argument('--num_epoch',          default=300, type=int)
    parser.add_argument('--valid_gap',          default=10, type=int)
    parser.add_argument('--preview_gap',        default=100, type=int)
    parser.add_argument('--checkpoint_gap',     default=20, type=int)
    parser.add_argument('--dump_loss_gap',      default=-1, type=int)
    
    parser.add_argument('--half',               default=False, type=bool)
    parser.add_argument('--num_workers',        default=12, type=int)
    parser.add_argument('--device',             default='cuda:0', type=str)
    
    parser.add_argument('--comment',            default='test', type=str)
    
    parser.add_argument('--checkpoint',         default=None, type=str)
    parser.add_argument('--dump_path',          default='dump', type=str)
    
    parser.add_argument('--pathcsi',            default='annotate/csi/data/lxh.pk', type=str)
    parser.add_argument('--pathmask',           default='annotate/maskrcnn/data/lxh.pk', type=str)
    parser.add_argument('--pathpose',           default='annotate/openpose/data/lxh.pk', type=str)
    
    parser.add_argument('--testfrac',           default=0.05, type=float)
    parser.add_argument('--trainfrac',          default=0.80, type=float)
    parser.add_argument('--eventlength',        default=100, type=float)
    
    parser.add_argument('--loadermode',         default='framewise', type=str)
    
    return parser