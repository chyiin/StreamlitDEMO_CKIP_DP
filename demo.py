
import networkx as nx
import numpy as np
import streamlit as st
from PIL import Image
import rpyc
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
from nltk.tokenize import word_tokenize
from transformers import XLNetTokenizer, BertTokenizer
from XLNet_encoder import XLNet_Encoder1
from XLNet_semantic import XLNet_Semantic
from BERT_encoder import BERT_Encoder1
from BERT_semantic import BERT_Semantic
import matplotlib.pyplot as plt
from opencc import OpenCC
from mst import mst_parse
from PIL import Image

s2t = OpenCC('s2tw')
t2s = OpenCC('t2s')
nltk.download('punkt')   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.cuda.set_device(0)

def tokenize_and_preserve_labels(tokenizer, subsent):
    
    tokenized_sentence, labels_idx, seq = [], [], [0]
    n, idx = 0, 0
    
    for word in subsent:

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        n = n + n_subwords
        tokenized_sentence.extend(tokenized_word)
        labels_idx.extend([idx] * n_subwords)
        seq.append(n)
        idx = idx + 1
        
    return tokenized_sentence, torch.tensor(labels_idx), seq[:-1]

@st.cache(allow_output_mutation=True)
class english_parser():

    def __init__(self):

        self.ids_to_labels = {0: 'root', 1: 'prep', 2: 'det', 3: 'nn', 4: 'num', 5: 'pobj', 6: 'punct', 7: 'poss', 8: 'possessive', 9: 'amod', 10: 'nsubj', 11: 'appos', 12: 'dobj', 13: 'dep', 14: 'cc', 15: 'conj', 16: 'nsubjpass', 17: 'partmod', 18: 'auxpass', 19: 'advmod', 20: 'ccomp', 21: 'aux', 22: 'cop', 23: 'xcomp', 24: 'quantmod', 25: 'tmod', 26: 'neg', 27: 'infmod', 28: 'rcmod', 29: 'pcomp', 30: 'mark', 31: 'advcl', 32: 'predet', 33: 'csubj', 34: 'mwe', 35: 'parataxis', 36: 'npadvmod', 37: 'number', 38: 'acomp', 39: 'prt', 40: 'iobj', 41: 'preconj', 42: 'expl', 43: 'discourse', 44: 'csubjpass'}

        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        self.model = XLNet_Encoder1(hidden_size=1024, pretrained="xlnet-large-cased").cuda()
        self.model.load_state_dict(torch.load(f'demo_model/xlnet_encoder.pt')) 

        self.semantic_model = XLNet_Semantic(loaded_model='demo_model/xlnet_encoder.pt', hidden_size=1024, num_labels=len(self.ids_to_labels)).cuda()
        self.semantic_model.load_state_dict(torch.load(f'demo_model/xlnet_punct_encoder.pt')) 

    def output(self, input_text):

        ws_token = word_tokenize(input_text)
        input_sentence = ['root'] + ws_token
        sentence_index = [(i, input_sentence[i]) for i in range(len(input_sentence))]

        input_token, input_idx, input_seqs = tokenize_and_preserve_labels(self.tokenizer, input_sentence)
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(txt) for txt in input_token])
        attention_masks = torch.tensor([float(i != 0.0) for i in input_ids])

        self.model.eval()
        self.semantic_model.eval()

        with torch.no_grad():
            
            output = self.model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())
            semantic_output = self.semantic_model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())

            semantic_label_indices = np.argmax(semantic_output[1].to('cpu').numpy(), axis=2)
            seq_output = torch.index_select(output[1][0], 0, torch.tensor(input_seqs).cuda())
            seq_output = torch.index_select(seq_output, 1, torch.tensor(input_seqs).cuda())
            label_indices = mst_parse(torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)

            final_predict = [int(input_idx[input_seqs[label_idx]].cpu()) for label_idx in label_indices]
            final_predict[0] = -1

            semantic_labels = [self.ids_to_labels[sem_idx] for sem_idx in semantic_label_indices[0]]
            semantic_predict = [semantic_labels[idx] for idx in input_seqs]

        parse = []
        for i in range(len(ws_token)):
            if final_predict[1:][i] == 0:
                parse.append((final_predict[1:][i], i+1, 'root'))
            else:
                parse.append((final_predict[1:][i], i+1, semantic_predict[1:][i]))

        return parse, sentence_index

@st.cache(allow_output_mutation=True)
class chinese_parser():

    def __init__(self):
        
        self.conn = rpyc.classic.connect('localhost', port=3333)
        self.conn.execute('from ckiptagger import data_utils, construct_dictionary, WS, POS, NER')
        self.conn.execute('ws = WS("./data")')
        self.ids_to_labels = {0: 'root', 1: 'nn', 2: 'conj', 3: 'cc', 4: 'nsubj', 5: 'dep', 6: 'punct', 7: 'lobj', 8: 'loc', 9: 'comod', 10: 'asp', 11: 'rcmod', 12: 'etc', 13: 'dobj', 14: 'cpm', 15: 'nummod', 16: 'clf', 17: 'assmod', 18: 'assm', 19: 'amod', 20: 'top', 21: 'attr', 22: 'advmod', 23: 'tmod', 24: 'neg', 25: 'prep', 26: 'pobj', 27: 'cop', 28: 'dvpmod', 29: 'dvpm', 30: 'lccomp', 31: 'plmod', 32: 'det', 33: 'pass', 34: 'ordmod', 35: 'pccomp', 36: 'range', 37: 'ccomp', 38: 'xsubj', 39: 'mmod', 40: 'prnmod', 41: 'rcomp', 42: 'vmod', 43: 'prtmod', 44: 'ba', 45: 'nsubjpass'}
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.model = BERT_Encoder1(hidden_size=1024, pretrained="hfl/chinese-roberta-wwm-ext-large").cuda()
        self.model.load_state_dict(torch.load(f'demo_model/bert_encoder.pt')) 

        self.semantic_model = BERT_Semantic(loaded_model='demo_model/bert_encoder.pt', hidden_size=1024, num_labels=len(self.ids_to_labels)).cuda()
        self.semantic_model.load_state_dict(torch.load(f'demo_model/bert_punct_encoder.pt')) 

    def output(self, input_text):

        ws_sent = self.conn.eval(f'ws(["{input_text}"])')[0]
        st2_sent = ['root'] + list(ws_sent)
        t2s_sent = ['root'] + [t2s.convert(word) for word in ws_sent]
        parse_input = ['root'] + list(ws_sent)
        sentence_index = [(i, st2_sent[i]) for i in range(len(st2_sent))]

        input_token, input_idx, input_seqs = tokenize_and_preserve_labels(self.tokenizer, t2s_sent)
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(txt) for txt in input_token])
        attention_masks = torch.tensor([float(i != 0.0) for i in input_ids])
        
        self.model.eval()
        self.semantic_model.eval()

        with torch.no_grad():
            
            output = self.model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())
            semantic_output = self.semantic_model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())

            semantic_label_indices = np.argmax(semantic_output[1].to('cpu').numpy(), axis=2)
            seq_output = torch.index_select(output[1][0], 0, torch.tensor(input_seqs).cuda())
            seq_output = torch.index_select(seq_output, 1, torch.tensor(input_seqs).cuda())
            label_indices = mst_parse(torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)

            final_predict = [int(input_idx[input_seqs[label_idx]].cpu()) for label_idx in label_indices]
            final_predict[0] = -1

            semantic_labels = [self.ids_to_labels[sem_idx] for sem_idx in semantic_label_indices[0]]
            semantic_predict = [semantic_labels[idx] for idx in input_seqs]

        parse = []
        for i in range(len(ws_sent)):
            if final_predict[1:][i] == 0:
                parse.append((final_predict[1:][i], i+1, 'root'))
            else:
                parse.append((final_predict[1:][i], i+1, semantic_predict[1:][i]))
  
        return parse, sentence_index

if __name__ == '__main__':

    st.set_page_config(
        page_title="Dependency Parser",
        page_icon=Image.open('ckip_logo.png'),
        layout="wide",
        initial_sidebar_state="expanded",
        )

    st.title('Dependency Parser')
    st.write("")

    option = st.selectbox('Select Language:', ('Chinese', 'English'))

    if option == 'Chinese':

        form = st.form(key='my_form')
        inp_text = form.text_input(label='請輸入句子 ...')
        submit_button = form.form_submit_button(label='確定')

        if inp_text == '':
            input_text = '你好嗎？'
        else:
            input_text = inp_text

        st.write("")
        st.write("剖析樹: ")

        with st.spinner('In progress ...'):
            
            output, indexx = chinese_parser().output(input_text)
            g=nx.DiGraph()
            for ind in indexx:
                g.add_node(ind[0], label=f'<{ind[1]}>')
            for tok_ind in output:
                g.add_edge(tok_ind[0], tok_ind[1], label=f' {tok_ind[2]} ')
            p=nx.drawing.nx_pydot.to_pydot(g)
            st.graphviz_chart(f"{p}", use_container_width=True)

    else:

        form = st.form(key='my_form')
        inp_text = form.text_input(label='Please submit sentence ...')
        submit_button = form.form_submit_button(label='Submit')

        if inp_text == '':
            input_text = 'How are you?' # 'Not all those who wrote appose the changes.'
        else:
            input_text = inp_text

        st.write("")
        st.write("Dependency Parse Tree: ")

        with st.spinner('In progress ...'):
        
            output, indexx = english_parser().output(input_text)
            g=nx.DiGraph()
            for ind in indexx:
                g.add_node(ind[0], label=f'<{ind[1]}>')
            for tok_ind in output:
                g.add_edge(tok_ind[0], tok_ind[1], label=f' {tok_ind[2]} ')
            p=nx.drawing.nx_pydot.to_pydot(g)
            st.graphviz_chart(f"{p}", use_container_width=True)