# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:46:37 2021

@author: sun73
"""

def get_txt_data(input_path):
    f = open(input_path, 'r')
    data = f.readlines()
    f.close()
    return data