#!/usr/bin/env python
from __future__ import print_function

import argparse
import datetime
import os
import time
import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml

def get_parser():
    parser = argparse.ArgumentParser(
        description='GADDCCANet')
    parser.add_argument(
        '--config',
        default='./config/run.yaml',
        help='path to the configuration file')

    parser.add_argument(
        '--data-feeder-args',
        default=dict(),
        help='data loader')


    # model
    parser.add_argument(
        '--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for running')

    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""
    def __init__(self, arg):
        self.arg = arg
        self.load_model()
        self.loadAllData(**self.arg.data_feeder_args)

    def loadAllData(self, data_path_view1, data_path_view2, label_path, testdata_path_view1, testdata_path_view2, testlabel_path):
        self.label = np.load(label_path)
        self.testlabel = np.load(testlabel_path)
        self.data_view1 = np.load(data_path_view1)
        self.data_view2 = np.load(data_path_view2)
        self.testdata_view1 = np.load(testdata_path_view1)
        self.testdata_view2 = np.load(testdata_path_view2)


    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        print(Model)
        self.model = Model().cuda(output_device)


    def start(self):        
        print('start running')
        startTime = datetime.datetime.now()
        with torch.no_grad():
            data_view1 = torch.from_numpy(self.data_view1).float().cuda(self.output_device)
            data_view2 = torch.from_numpy(self.data_view2).float().cuda(self.output_device)
            label = torch.from_numpy(self.label).long().cuda(self.output_device)
            testdata_view1 = torch.from_numpy(self.testdata_view1).float().cuda(self.output_device)
            testdata_view2 = torch.from_numpy(self.testdata_view2).float().cuda(self.output_device)
            testlabel = torch.from_numpy(self.testlabel).long().cuda(self.output_device)
            self.model(data_view1, data_view2, label, testdata_view1, testdata_view2, testlabel)
        endTime = datetime.datetime.now()
        print('done')
        print('runing time:' + str(endTime-startTime))

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()
    torch.set_printoptions(threshold=np.inf)
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    processor = Processor(arg)
    processor.start()
