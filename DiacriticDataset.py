#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch
import json
import pickle
import string
import pyarabic.araby as araby
import pyarabic.number as number


# In[78]:


class DiacriticDataset(Dataset):
    def __init__(self,dataset_path,letter_to_id_path,id_to_letter_path,diacritic_to_id_path,id_to_diacritic_path,word_to_id_path,id_to_word_path):
        
        self.file = dataset_path
        
        letter_to_id_file= open(letter_to_id_path, 'rb')
        self.letter_to_id = pickle.load(letter_to_id_file)
        letter_to_id_file.close()
        
        id_to_letter_file = open(id_to_letter_path, 'rb')
        self.id_to_letter = pickle.load(id_to_letter_file)
        id_to_letter_file.close()
        
        diacritic_to_id_file= open(diacritic_to_id_path, 'rb')
        self.diacritic_to_id = pickle.load(diacritic_to_id_file)
        diacritic_to_id_file.close()
        
        id_to_diacritic_file = open(id_to_diacritic_path, 'rb')
        self.id_to_diacritic = pickle.load(id_to_diacritic_file)
        id_to_diacritic_file.close()
        
        word_to_id_file= open(word_to_id_path, 'rb')
        self.word_to_id = pickle.load(word_to_id_file)
        word_to_id_file.close()
        
        id_to_word_file = open(id_to_word_path, 'rb')
        self.id_to_word = pickle.load(id_to_word_file)
        id_to_word_file.close()
        
        self.data = self.prepare_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    
    def prepare_dataset(self):
        data = {}
        counter = 0
        with open(self.file,encoding='utf8') as file:
            for line in file:
                letter_ids = []
                diacritic_ids = []
                word_ids = []
                letters, diacritics =araby.separate(line)
                letters = letters[0:-1]
                words = araby.tokenize(line)[0:-1]
                diacritics = diacritics[0:-1]
                for letter in letters:
                    if (letter == '\n') or (letter == '\u200f'):
                        continue
                        
                    letter_ids.append(self.letter_to_id[letter])
                    
                for index,diacritic in enumerate(diacritics):
                    if letters[index] == " ":
                        diacritic_ids.append(self.diacritic_to_id['space'])
                    else:
                        diacritic_ids.append(self.diacritic_to_id[diacritic])
                
                for word in words:
                    word_ids.append(self.word_to_id[araby.strip_tashkeel(word)])
                
                instance = (torch.tensor(letter_ids,dtype=torch.long,requires_grad=False),
                           torch.tensor(diacritic_ids,dtype=torch.long,requires_grad=False),
                           torch.tensor(word_ids,dtype=torch.long,requires_grad=False))
                data[counter] = instance
                counter += 1
        return data

class DiacriticDatasetShaddah(Dataset):
    def __init__(self,dataset_path,letter_to_id_path,id_to_letter_path,diacritic_to_id_path,id_to_diacritic_path,word_to_id_nodiacs_path,                   id_to_word_nodiacs_path,diacritic_to_id_nosh_path,id_to_diacritic_nosh_path,word_to_id_diacs_path,id_to_word_diacs_path):
        
        self.file = dataset_path
        
        letter_to_id_file= open(letter_to_id_path, 'rb')
        self.letter_to_id = pickle.load(letter_to_id_file)
        letter_to_id_file.close()
        
        id_to_letter_file = open(id_to_letter_path, 'rb')
        self.id_to_letter = pickle.load(id_to_letter_file)
        id_to_letter_file.close()
        
        diacritic_to_id_file= open(diacritic_to_id_path, 'rb')
        self.diacritic_to_id = pickle.load(diacritic_to_id_file)
        diacritic_to_id_file.close()
        
        id_to_diacritic_file = open(id_to_diacritic_path, 'rb')
        self.id_to_diacritic = pickle.load(id_to_diacritic_file)
        id_to_diacritic_file.close()
        
        word_to_id_nodiacs_file= open(word_to_id_nodiacs_path, 'rb')
        self.word_to_id_nodiacs = pickle.load(word_to_id_nodiacs_file)
        word_to_id_nodiacs_file.close()
        
        id_to_word_nodiacs_file = open(id_to_word_nodiacs_path, 'rb')
        self.id_to_word_nodiacs = pickle.load(id_to_word_nodiacs_file)
        id_to_word_nodiacs_file.close()
        
        diacritic_to_id_nosh_file= open(diacritic_to_id_nosh_path, 'rb')
        self.diacritic_to_id_nosh = pickle.load(diacritic_to_id_nosh_file)
        diacritic_to_id_nosh_file.close()
        
        id_to_diacritic_nosh_file = open(id_to_diacritic_nosh_path, 'rb')
        self.id_to_diacritic_nosh = pickle.load(id_to_diacritic_nosh_file)
        id_to_diacritic_nosh_file.close()
        
        word_to_id_diacs_file= open(word_to_id_diacs_path, 'rb')
        self.word_to_id_diacs = pickle.load(word_to_id_diacs_file)
        word_to_id_diacs_file.close()
        
        id_to_word_diacs_file = open(id_to_word_diacs_path, 'rb')
        self.id_to_word_diacs = pickle.load(id_to_word_diacs_file)
        id_to_word_diacs_file.close()
        
        
        self.data = self.prepare_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
    
    def prepare_dataset(self):
        data = {}
        counter = 0
        with open(self.file,encoding='utf-8') as file:
            for line in file:
                letter_ids = []
                diacritic_ids = []
                word_ids_nodiacs = []
                word_ids_diacs = []
                letters, diacritics =araby.separate(line)
                letters = letters[0:-1]
                words = araby.tokenize(line)[0:-1]
                diacritics = diacritics[0:-1]
                diacritic_ids_nosh = []
                index = 0
                shaddahs = []
                for letter in letters:
                    if (letter == '\n') or (letter == '\u200f'):
                        continue
                    if (letter =='ّ'):
                        if diacritics[index] != 'ـ':
                            diacritic_ids[-1] = self.diacritic_to_id[letter+diacritics[index]]
                            
                        else:
                            diacritic_ids[-1] = self.diacritic_to_id[letter]
                            
                        diacritic_ids_nosh[-1] = self.diacritic_to_id_nosh[diacritics[index]]
                        
                            
                    else:
                        letter_ids.append(self.letter_to_id[letter])
                        if letter == " ":
                            diacritic_ids.append(self.diacritic_to_id['space'])
                            diacritic_ids_nosh.append(self.diacritic_to_id_nosh['space'])
                            
                        else:
                            diacritic_ids.append(self.diacritic_to_id[diacritics[index]])
                            diacritic_ids_nosh.append(self.diacritic_to_id_nosh[diacritics[index]])
                            
                    index += 1
                
                
                
                for diacritic_id in diacritic_ids:
                    if 'ّ' in self.id_to_diacritic[diacritic_id]:
                        shaddahs.append(1)
                    else:
                        shaddahs.append(0)
                    
                for word in words:
                    word_ids_diacs.append(self.word_to_id_diacs[word])
                    word_ids_nodiacs.append(self.word_to_id_nodiacs[araby.strip_tashkeel(word)])
                    

                instance = (torch.tensor(letter_ids,dtype=torch.long,requires_grad=False),
                           torch.tensor(diacritic_ids,dtype=torch.long,requires_grad=False),
                           torch.tensor(diacritic_ids_nosh,dtype=torch.long,requires_grad=False),
                           torch.tensor(shaddahs,dtype=torch.long,requires_grad=False),
                           torch.tensor(word_ids_diacs,dtype=torch.long,requires_grad=False),
                           torch.tensor(word_ids_nodiacs,dtype=torch.long,requires_grad=False))
                data[counter] = instance
                counter += 1
        return data