# -*- coding: utf-8 -*-


import os
import torch
import numpy as np
import itertools
from typing import List
from tqdm import tqdm
import argparse
import pickle
import torch.nn.functional as F

from evaluate_rxn_yield import get_classes
from train_multilabel.rxn_model_morgan import ReactionModel_Listwise
from train_multilabel.args_train import TrainArgs_rxn

def one_hot(context):
    """Convert the gold answer solvent ans reagent names to the one-hot feature vector. """
    global solvent_classes
    global reagent_classes
    solvent, reagent = context
    solvent = solvent.split('; ')
    vec_solv = np.array([float(x[0] in solvent) for x in solvent_classes])
    reagent = reagent.split('; ')
    vec_reag = np.array([float(x[0] in reagent) for x in reagent_classes])
    
    return vec_solv, vec_reag

def convert_contexts2tensor(contexts):
    solvent_batch = []
    reagent_batch = []
    for context in contexts:
        vec_solv, vec_reag = one_hot(context)
        solvent_batch.append(vec_solv)
        reagent_batch.append(vec_reag)
    
    return torch.Tensor(solvent_batch), torch.Tensor(reagent_batch)

def get_answer(rxn):
    """Get the condition gold answer in the dataset """
    answers = []
    for cond in rxn[3]:
        reagent = cond[1]; solvent = cond[2];
        reagent = set(reagent.split('; '))
        solvent = set(solvent.split('; '))
        answers.append((solvent, reagent))
    return answers

def compare_answer_and_combinations(answers, context_combinations):
    for answer in answers:
        answer_s, answer_r = answer
        for i, context in enumerate(context_combinations):
            solvent, reagent = context
            solvent = set(solvent.split('; '))
            reagent = set(reagent.split('; '))
            if (answer_s == solvent) & (answer_r == reagent):
                return i
    return None

def evaluate_overall(acc_list, show=(1,3,5,10,15,20,25)):
    topk_dict = dict(zip(show,[0]*len(show)))
    length = len(acc_list)
    for rank in acc_list:
        if rank == None: continue
        for key in topk_dict.keys():
            if rank <= key:
                topk_dict[key] += 1
    
    for key, value in topk_dict.items():
        print("top accuracy@{} : {:.4f}".format(key, value/length))
    return

def make_dummy(solvent_classes, reagent_classes, topk=5):
    solv = solvent_classes[:topk]
    reag = reagent_classes[:topk]
    dummy_list = list(itertools.product(solv, reag))
    dummy_list = [(x[0][0], x[1][0], x[0][1]+x[1][1]) for x in dummy_list]
    dummy_list = sorted(dummy_list, key=lambda x:x[2],reverse=True)
    return [(x[0],x[1]) for x in dummy_list]


if __name__ == '__main__':
    
    """
    Get the reaction data, the solvent classes and the reagent classes.
    """
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(os.path.join(os.path.join(data_path, 'All_LCC_Data'), 'processed_data'), '05Final_for_second_part_model')
    
    solvent_classes = get_classes(os.path.join(data_path, 'class_names_solvent.pkl'))
    reagent_classes = get_classes(os.path.join(data_path, 'class_names_reagent.pkl'))
    f = open(os.path.join(data_path, 'combinations_of_validate.pkl'), 'rb')
    data = pickle.load(f)
    f.close()

    dummy_list = make_dummy(solvent_classes,reagent_classes, topk=5)
    """
    Start to evaluate the ranking of the different context combinations.
    """
    acc_list = []
    for i, rxn in enumerate(data[:]):
        gold_answers = get_answer(rxn)
        
        id_ = compare_answer_and_combinations(gold_answers, dummy_list)
        if id_ == None:
            print(i)
        acc_list.append(id_)
    
    evaluate_overall(acc_list)