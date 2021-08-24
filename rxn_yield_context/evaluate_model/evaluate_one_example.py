# -*- coding: utf-8 -*-

import os
import torch
import argparse

from rxn_yield_context.train_multilabel.data_utils import get_classes, create_rxn_Morgan2FP_concatenate
from rxn_yield_context.evaluate_model.eval_utils import MultiTask_Evaluator



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir',type=str,
                        default='../All_LCC_Data/processed_data_12')
    parser.add_argument('--multitask_model',type=str,
                        default='../save_model/first_model_12_final_3/multitask_model_epoch-100.checkpoint')
    parser.add_argument('--cutoff_solvent',type=float,default=0.3)
    parser.add_argument('--cutoff_reagent',type=float,default=0.25)
    args_r = parser.parse_args()
    
    solvent_classes = get_classes(os.path.join(os.path.join(args_r.test_dir, 'label_processed'), 'class_names_solvent_labels_processed.pkl'))
    reagent_classes = get_classes(os.path.join(os.path.join(args_r.test_dir, 'label_processed'), 'class_names_reagent_labels_processed.pkl'))
    
    MT_Evaluator = MultiTask_Evaluator(solvent_classes, reagent_classes, cutoff_solv= args_r.cutoff_solvent, cutoff_reag = args_r.cutoff_reagent)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    MT_Evaluator.load_model(args_r.multitask_model, device = device)
    
    
    rxn_smiles = 'BrC1=CC=C(CN2C=NN=N2)C=C1.CC(=O)C1=CC=C(C=C1)B(O)O>>CC(=O)C1=CC=C(C=C1)C1=CC=C(CN2C=NN=N2)C=C1'
    
    reac, prod = rxn_smiles.split('>>')
    rxn_fp = torch.Tensor(create_rxn_Morgan2FP_concatenate(reac, prod, fpsize = MT_Evaluator.args_MT.fpsize, radius = MT_Evaluator.args_MT.radius))
    rxn_fp = rxn_fp.to(device)
    
    MT_Evaluator.predict_context(rxn_fp, verbose = True)