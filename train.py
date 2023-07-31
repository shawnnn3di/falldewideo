from utils import include_args

import argparse

def train():
    
    args = include_args()
    
    device = args.device
    commentstr = args.commentstr