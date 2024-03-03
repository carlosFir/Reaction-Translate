import torch
import pdb
from torch.utils.data import DataLoader, Dataset 



class Tokenizer():
    '''
    vocab is a dictionary with keys as the words and values as the index
    '''
    def __init__(self, vocab_txt):

        self.vocab = self.build_vocab(vocab_txt)
        self.word2index = self.vocab
        self.index2word = list(self.vocab.keys())
        self.vocab_size = len(self.vocab)

        # char level tokenizer is banned
        # self.split_char = lambda x: [char for char in x]

        # atom level tokenizer is used
        self.split_atom = self.smi_tokenizer

    def build_vocab(self, vocab_txt):
        '''
        vocab_txt is the path to the vocab.txt including tokens

        return a dictionary with keys as the words and values as the index
        '''

        # init slow version
        result_dict = {}
        with open(vocab_txt, 'r') as file:
            for line_number, line in enumerate(file):
                line = line.strip()
                result_dict[line] = line_number
        return result_dict
    
    def encode(self, sentence, max_length):
        '''
        sentence is a string, without any split or tokenization, e.g. O=[N+]([O-])c1ccc(N=C2NC3(CCCC3)CS2)c2c1CCCC2
        '''

        # sentence = self.split_char(sentence)
        # sentence = 'O=[N+]([O-])c1ccc(N=C2NC3(CCCC3)CS2)c2c1CCCC2'
        sentence = self.split_atom(sentence)
        # sentence: 'O = [N+] ( [O-] ) c 1 c c c ( N = C 2 N C 3 ( C C C C 3 ) C S 2 ) c 2 c 1 C C C C 2'
        sentence = sentence.split() # to list
        sentence = ['<start>'] + sentence + ['<end>']
        sentence_ids = [self.word2index[word] for word in sentence]

        pad_ids = self.word2index['<pad>']
        sentence_ids.extend([pad_ids] * max_length)

        return sentence_ids[:max_length]
    
        
    def decode(self, sentence_ids):
        # it has not been used yet...
        
        sts = ''
        for ids in sentence_ids:
            if ids >= self.vocab_size:
                sts += '<unk>'
                continue
            
            word = self.index2word[int(ids)]
            sts += word
        
        return sts
    
    def smi_tokenizer(self, smi):
        """
        Tokenize a SMILES molecule or reaction
        """
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
    
class ReaDataset(Dataset):
    '''
    data is SMLIES list like [[src, targ], [...], ...]
    vocab_file is the path to the vocab.txt
    max_length is the max length of the sentence
    '''
    def __init__(self, data, vocab_file, max_length):
        super(ReaDataset, self).__init__()
        
        self.init_data = data
        self.tokenizer = Tokenizer(vocab_file)
        self.data = self.pad_and_token(data, max_length)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def pad_and_token(self, data, max_length):

        result = [None] * len(data)  # Preallocate result list
        for i, pair in enumerate(data):
            src = self.tokenizer.encode(pair[0], max_length)  # Encode source sentence
            targ = self.tokenizer.encode(pair[1], max_length)  # Encode target sentence
            result[i] = [torch.tensor(src), torch.tensor(targ)]  # Store tensors in result list
        return result

if __name__ == '__main__':
    data = [['Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>',
             'O=[N+]([O-])c1ccc(N=C2NC3(CCCC3)CS2)c2c1CCCC2'],
             ['c2c1CCCC2>',
             '(CCCC3)CS2)c2c1CCCC2']]
    
    dataset = ReaDataset(data, 'vocab.txt', 100)
   
    dataloader = DataLoader(dataset, 1)
    for batch in dataloader:
        src, targ = batch[0], batch[1]
        print(src, targ.shape)
        break

        
