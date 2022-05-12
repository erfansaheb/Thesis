# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:08:03 2022

@author: erfan
"""
import xmltodict

def load_xml(file):
    # load file
    with open(file, 'r') as f:
        data_dict = xmltodict.parse(f.read())
    #Capacity constraints
    try:
        ca = data_dict['Instance']['Constraints']['CapacityConstraints']
    except:
        ca = None 
    #Game Constraints
    try:
        ga = data_dict['Instance']['Constraints']['GameConstraints']
    except:
        ga = None
    #Break Constraints
    try:
        ba = data_dict['Instance']['Constraints']['BreakConstraints']
    except:
        ba = None
    #Fairness Constraints
    try:
        fa = data_dict['Instance']['Constraints']['FairnessConstraints']
    except:
        fa = None
    #Separation Constraints
    try:
        sa = data_dict['Instance']['Constraints']['SeparationConstraints']
    except:
        sa = None
        
    return ca, ga, ba, fa, sa

ca, ga, ba, fa, sa = load_xml('Instances//LateInstances//ITC2021_Late_1.xml')