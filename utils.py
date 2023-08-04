import argparse


def include_args(parser: argparse.ArgumentParser):
    '''
    training args:
        shuffle_train                   where training set is shuffled
        batch_size                      batch size
        lr                              learning rate
        weight_decay                    a parameter for Adam optimizer
        lr_gamma                        decay speed for learning rate. only activated when there is 
                                            a scheduler in training
        num_epoch                       how many epochs you run the data loader
        valid_gap                       how often you validate your model during training
        preview_gap                     how often you dump results in training epochs
        checkpoint_gap                  how often you checkpoint the model
        half                            train in float16 instead of float32
        num_workers                     how many subprocesses used for data loading
        device                          which device you use, cpu, cuda:0 or cuda:1
        comment                         discriminate different runs
        checkpoint                      path to pretrained models when finetuning
        dump_path                       where you dump your results
        pathcsi                         where you store your packed CSI data
        pathmask                        where you store your packed mask labels
        pathpose                        where you store your packed pose labels
        testfrac                        percentage of test set
        trainfrac                       percentage of train set
        samplelength                    how many consecutive frames the model take in at the same time
        eventlength                     in event-level data separation scheme, how long is the event
        loadermode                      model data framewise or eventwise. 'eventwise' not implemented yet.
    '''
    
    parser.add_argument('--shuffle_train',      default=True, type=bool)
    parser.add_argument('--batch_size',         default=32, type=int)
    parser.add_argument('--lr',                 default=1e-3, type=float)
    parser.add_argument('--weight_decay',       default=1e-5, type=float)
    parser.add_argument('--lr_gamma',           default=0.99, type=float)
    parser.add_argument('--num_epoch',          default=300, type=int)
    parser.add_argument('--valid_gap',          default=20, type=int)
    parser.add_argument('--preview_gap',        default=100, type=int)
    parser.add_argument('--checkpoint_gap',     default=20, type=int)
    
    parser.add_argument('--half',               default=False, type=bool)
    parser.add_argument('--num_workers',        default=12, type=int)
    parser.add_argument('--device',             default='cuda:0', type=str)
    
    parser.add_argument('--comment',            default='test', type=str)
    
    parser.add_argument('--checkpoint',         default=None, type=str)
    parser.add_argument('--dump_path',          default='dump', type=str)
    parser.add_argument('--traindumpgap',       default=999, type=int)
    
    parser.add_argument('--pathcsi',            default='annotate/csi/data/lxh.pk', type=str)
    parser.add_argument('--pathmask',           default='annotate/maskrcnn/data/lxh.pk', type=str)
    parser.add_argument('--pathpose',           default='annotate/openpose/data/lxh.pk', type=str)
    
    parser.add_argument('--testfrac',           default=0.05, type=float)
    parser.add_argument('--trainfrac',          default=0.80, type=float)
    parser.add_argument('--samplelength',       default=1, type=int)
    parser.add_argument('--eventlength',        default=100, type=int)
    
    parser.add_argument('--loadermode',         default='framelevel', type=str)
    
    return parser