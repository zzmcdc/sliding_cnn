#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:25:52 2018

@author: zhao
"""
import json
import yaml
class train_param(object):
    def __init__(self):
        self.parm = {}
        
    def load_parm(self,yaml_file):
        file = open(yaml_file,'r')
        self.parm = yaml.load(file)
        file.close()
        for item in self.parm:
            print(item,self.parm[item])
            
    def get_parm(self):
        return self.parm
    def __str__(self):
        return self.parm

if __name__=='__main__':
    parm = train_param()
    parm.load_parm('config.yaml')
    test = parm.get_parm()
    print(test)
