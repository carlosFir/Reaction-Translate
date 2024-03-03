from .module import *
import torch
from .loader import *
import pdb
from queue import PriorityQueue
import operator

def greedy_decoder(model, encoder_input, tokenizer, MAX_LENGTH, pad):
    '''
    inp_sentence_ids: list of int
    tokenizer: instance of class Tokenizer()

    return the output of greedy decoder with taecher forcing
    '''
    model.eval()
    tk = Tokenizer('vocab.txt')
    decoder_input = torch.tensor([tk.word2index['<start>']]).unsqueeze(0).to(encoder_input.device)

    # print(encoder_input.device, decoder_input.device)
    with torch.no_grad():

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, decoder_input, pad)
            # print(enc_padding_mask.device)

            predictions, _ = model(encoder_input, 
                                   decoder_input, 
                                   enc_padding_mask, 
                                   combined_mask, 
                                   dec_padding_mask)
            # print(predictions)
            next_word = predictions[:, -1, :]
            next_ids = torch.argmax(next_word, dim=-1)
            # print(next_ids.shape, decoder_input.shape)
            
            if next_ids.squeeze().item() == tokenizer.word2index['<end>']:
                return decoder_input.squeeze(0), _
            
            decoder_input = torch.cat([decoder_input, next_ids.unsqueeze(0)], dim=1)
            
    return decoder_input.squeeze(0), _

    # print(encoder_input.shape, decoder_input)


class BeamSearchNode(object):
    def __init__(self, previousNode, wordID, logProb, length):
        self.prevNode = previousNode
        self.wordid = wordID
        self.logp = logProb
        self.leng = length
    
    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(model, encoder_input, tokenizer, beam_width, topk, MAX_LENGTH, pad):
    
    decoded_batch = []
    # decoding goes sentence by sentence
    for idx in range(encoder_input.shape[0]):

        # initialize the first node: start node
        # initialize the nodes queue, endnodes, qsize
        
        decoder_input = torch.LongTensor([[tokenizer.word2index['<start>']]]).to(encoder_input.device) # (1, 1)

        endnodes = []
        # number_required = min((topk + 1), topk - len(endnodes)) # why?
        number_required = topk

        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node)) # (scores, node)
        qsize = 1

        while True:
            # print(idx, qsize)
            # there are two conditions to break the loop
            # 1. qsize is enough
            # 2. endnodes' length is enough
            if qsize > 2000:
                break

            score, n = nodes.get()
            decoder_input = n.wordid
            
            # end token and not the first token
            if n.wordid.item() == tokenizer.word2index['<end>'] and n.prevNode != None:
                print("End token generated")
                endnodes.append((score, n))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            
            e, c, d = create_mask(encoder_input, decoder_input, pad)
            decoder_output, _ = model(encoder_input, decoder_input, e, c, d) # decoder_input: (1, 1) 
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            indexes = indexes.squeeze(0)
            log_prob = log_prob.squeeze(0)  
            # print(indexes.shape, log_prob.shape)
            # 这里得到索引和概率是硬选择，而采样不会
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                # print(decoded_t)
                node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()

                nodes.put((score, node))
            qsize += beam_width - 1

        # if end token is not generated

        # show the nodes content(debug)
       
        '''while not nodes.empty():
            
            score, n = nodes.get()
            print(n.wordid)'''

        if len(endnodes) == 0:
            print("No end token generated")
            endnodes = [nodes.get() for _ in range(topk)]
            endnodes = sorted(endnodes, key=operator.itemgetter(0)) # ascending actually

        # pdb.set_trace()

        # utterances = torch.empty(0).to(encoder_input.device)
        utterances = []
        for score,n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = torch.empty(0).to(encoder_input.device)
            utterance = torch.cat([utterance, n.wordid], dim=1)
            '''utterance = []
            utterance.append(n.wordid)'''
            
            while n.prevNode != None:
                n = n.prevNode
                # utterance.append(n.wordid)
                utterance = torch.cat([utterance, n.wordid], dim=1)

            utterance = torch.flip(utterance, [1])
            utterances.append(utterance)

        decoded_batch.append(utterances)

    
    return decoded_batch


def beam_search(model, encoder_input, tokenizer, beam_size, MAX_LENGTH, pad):

    model.eval()
    tk = Tokenizer('vocab.txt')
    decoder_input = torch.tensor([tk.word2index['<start>']]).unsqueeze(0).to(encoder_input.device)


    with torch.no_grad():
        # init beam
        e, c, d = create_mask(encoder_input, decoder_input, pad)

        predictions, _ = model(encoder_input,
                                decoder_input,
                                e, c, d)
        
        next_word = predictions[:, -1, :]
        # softmax should be set with temperatue
        p, next_ids = torch.topk(torch.softmax(next_word, dim=-1), k=beam_size, dim=-1)
        
        
        encoder_input = encoder_input.repeat(2, 1)

        pdb.set_trace()

        print(encoder_input.shape)
    

        

    



def teacher_decoder(model, encoder_input, decoder_input, tokenizer, MAX_LENGTH, pad):
    '''
    inp_sentence_ids: list of int
    tokenizer: instance of class Tokenizer()

    return the output of greedy decoder with taecher forcing
    '''
    model.eval()

    # print(encoder_input.device, decoder_input.device)
    with torch.no_grad():

        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, decoder_input, pad)
        predictions, _ = model(encoder_input, 
                                decoder_input, 
                                enc_padding_mask, 
                                combined_mask, 
                                dec_padding_mask)
        predictions = predictions[:, :MAX_LENGTH, :]
        decoder_output = torch.argmax(predictions, dim=-1)
    
    return decoder_output

    # print(encoder_input.shape, decoder_input)
if __name__ == '__main__':
    
    sentence = 'CCCCSS'

    tk = Tokenizer('vocab.txt')
 

   



