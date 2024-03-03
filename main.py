import torchtext
from sklearn.model_selection import train_test_split
import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import configparser
import warnings
from torch.utils.data import DataLoader
import pdb
import pickle

warnings.filterwarnings('ignore')

from tool.loader import *
from tool.preprocess import *
from tool.module import *
from tool.generate import *

# setting
# ---------------config parsing---------------
config = configparser.ConfigParser()
config.read('config.conf')
# ---------------cuda and device----------------
device = torch.device(str(config['model']['device']) if torch.cuda.is_available() else "cpu")
# ---------------hyper parameters------------
# data
MAX_LENGTH = int(config['data']['max_length']) # truncated to fixed length
BATCH_SIZE = int(config['data']['batch_size'])
ON_FULL = str(config['data']['on_full']) == 'True'
if ON_FULL:
    data_path = str(config['data']['full_data_path'])
else:
    data_path = str(config['data']['250k_data_path'])
VOCAB_PATH = str(config['data']['vocab_path'])
# model
NUM_LAYERS = int(config['model']['num_layers'])
D_MODEL = int(config['model']['d_model'])
NUM_HEADS = int(config['model']['num_heads'])
DFF = int(config['model']['dff'])
DROPOUT = float(config['model']['rate'])
PE_INPUT = int(config['model']['pe_input'])
PE_TARGET = int(config['model']['pe_target'])
# train
EPOCHS = int(config['train']['epochs'])
SAVE_DIR = str(config['train']['save_dir'])
SAVE_NAME = str(config['train']['save_name'])
PRINT_TRAINSTEP_EVERY = int(config['train']['print_trainstep_every']) 
TEACHER_FORCING = float(config['train']['teacher_forcing_ratio'])
# optimizer
LR = float(config['optimizer']['lr'])
BETAS = (float(config['optimizer']['beta1']), float(config['optimizer']['beta2']))
EPS = float(config['optimizer']['eps'])  
# generate
GENERATE = str(config['generate']['on']) == 'True'
CKPT = str(config['generate']['ckpt'])

# -------------------------------Loading data-------------------------------
print("Loading....")
tk = Tokenizer(VOCAB_PATH)
input_vocab_size = tk.vocab_size
target_vocab_size = tk.vocab_size
pad = tk.word2index['<pad>']

# ---------------------------------Model--------------------------------

model = Transformer(num_layers=NUM_LAYERS,
                    d_model=D_MODEL,
                    num_heads=NUM_HEADS,
                    dff=DFF,
                    input_vocab_size=input_vocab_size,
                    target_vocab_size=target_vocab_size,
                    pe_input=PE_INPUT,
                    pe_target=PE_TARGET,
                    rate=DROPOUT).to(device)

print('Model parameters: ', count_parameters(model))

if not GENERATE: 

    with open('train_full.pkl', 'rb') as file:
        train_pairs = pickle.load(file)

    # Read test_pairs
    with open('test_full.pkl', 'rb') as file:
        test_pairs = pickle.load(file)

    print('Creating dataset...')
    train_dataset = ReaDataset(train_pairs, VOCAB_PATH, MAX_LENGTH)
    val_dataset = ReaDataset(test_pairs, VOCAB_PATH, MAX_LENGTH)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    print("Training....")
    with open('config.conf', 'r') as f:
        content = f.read()
        print(content)

    loss_object = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    df_history = train_model(model, 
                            optimizer,
                            EPOCHS, 
                            train_dataloader, 
                            val_dataloader, 
                            PRINT_TRAINSTEP_EVERY, 
                            SAVE_DIR, 
                            SAVE_NAME,
                            pad,
                            device,
                            TEACHER_FORCING)
    
else:
    with open('single_case.pkl', 'rb') as file:
        test_pairs = pickle.load(file)
    print('Creating dataset...')
    val_dataset = ReaDataset(test_pairs, VOCAB_PATH, MAX_LENGTH)
    val_dataloader = DataLoader(val_dataset, 1)

    ckpt = torch.load(CKPT)
    model_sd = ckpt['net']
    model.load_state_dict(model_sd)
    model.eval()


    for (inp, targ) in val_dataloader:
       
        encoder_input = inp[:, 1:].to(device)
        targ_output = targ[:, :-1].to(device)
        targ_real = targ[:, 1:].to(device)
        
        # 489 is end_ids
        # 1 is pad_ids after end_ids
        index = (targ_output[0] == 1).nonzero()

        if index.numel() > 0:
            pos = index[0].item()
        else:
            pos = len(targ_output[0])

        # 训练逻辑是与targ_real比较
        # tearcher_output是接近训练逻辑的解码方式
        # greedy_output是目前的validate的解码方式，无法准确评估
        
        
        '''greedy_output, _ = greedy_decoder(model, encoder_input, tk, pos, pad)
        greedy_output = greedy_output[1:].unsqueeze(0)
        teacher_output = teacher_decoder(model, encoder_input, targ_output, tk, pos, pad)'''

        out = beam_decode(model, encoder_input, tk, 10, 5, pos, pad)[0]


        for seq in out:
            seq = seq.squeeze(0)
            sm = tk.decode(seq)
            print(sm)
   
       
        break
        


        
        



    
    









