import os
INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
INTERACTIVE, os.environ['KAGGLE_KERNEL_RUN_TYPE'] 


import multiprocessing as mp
from multiprocessing import Pool    

import pandas as pd

from tqdm import tqdm 
from time import time
from datetime import date
today =date.today()

import pickle, json

import kag_persist_install

try:
       import wget
except:
       install_packages(['pip install wget'])
       import wget

if ('decoded_df2' in globals())==False:
       decoded_df2=pd.read_csv('/kaggle/input/neiss-sentence-transform-embeddings/decoded_df2__l1.csv')
       decoded_df2=decoded_df2.drop_duplicates('cpsc_case_number',)
       Embeddings={}
       
       decoded_df2.to_csv('decoded_df2_unique.csv')
       cpcs_nums = decoded_df2.cpsc_case_number
       size_n = decoded_df2.shape[0]

def embed( sentences ):             
    if TF_MODEL:
        #embeddings = model.predict( sentences )
        embeddings = model(sentences)
    else: # pytorch         
        embeddings = model.encode(sentences)            
    return embeddings

TF_MODEL = 0
#for src in [ 'narrative_cleaned','narrative' ]:    
for src in [ 'narrative' ]:    
    if INTERACTIVE:        
        inp = decoded_df2[src][:100] 
    else:
        inp =  decoded_df2[src] 
    
    if 'cleaned' not in src:        
        inp = inp.str.lower()
        
    inp = list(inp)
        
    for emb in EMB:
        starttime=time()
        if emb<4: 
            try:
                from sentence_transformers import SentenceTransformer
            except:                
                cmd=['pip install -U sentence-transformers']; install_packages( cmd ); 
                from sentence_transformers import SentenceTransformer
                
            if emb==1:# 768
                model = SentenceTransformer('all-mpnet-base-v2')
            elif emb==2: # 384
                model = SentenceTransformer('all-MiniLM-L6-v2')
            elif emb==3: #768
                model = SentenceTransformer('paraphrase-mpnet-base-v2')  # already explored paraphrase-multilingual-mpnet-base-v2")
        
        elif emb==4: # 512
            TF_MODEL = 1
            import tensorflow_hub as hub
            # Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. 
            # Universal Sentence Encoder. arXiv:1803.11175, 2018.
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"         
            model = hub.load(module_url)
            
        elif emb==5: # Roberta
            cmd=['pip install -U tensorflow==2.13']; install_packages( cmd ); import tensorflow as tf; import tokenization            
            #https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py            
            url='https://raw.githubusercontent.com/google-research/bert/master/tokenization.py';import wget; wget.download(url)
            
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.layers import Dense, Input

            def bert_encode(texts, tokenizer, max_len=512):
                all_tokens = []
                all_masks = []
                all_segments = []

                for text in texts:
                    text = tokenizer.tokenize(text)

                    text = text[:max_len-2]
                    input_sequence = ["[CLS]"] + text + ["[SEP]"]
                    pad_len = max_len - len(input_sequence)

                    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
                    tokens += [0] * pad_len
                    pad_masks = [1] * len(input_sequence) + [0] * pad_len
                    segment_ids = [0] * max_len

                    all_tokens.append(tokens)
                    all_masks.append(pad_masks)
                    all_segments.append(segment_ids)

                return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

            def build_model(bert_layer, max_len=512):

                input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
                input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
                segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

                #could be pooled_output, sequence_output yet sequence output provides for each input token (in context)
                _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
                clf_output = sequence_output[:, 0, :]
                out = Dense(1, activation='sigmoid')(clf_output)

                model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

                #specifying optimizer
                model.compile(Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

                return model

            module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
            bert_layer = hub.KerasLayer(module_url, trainable=True)
            model = build_model(bert_layer, )
            TF_MODEL = 1
            
        elif emb>=6:
            import tensorflow_hub as hub
            
            TF_MODEL = 1            
            if emb==6:
                model = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-small/1")
            elif emb==7:
                model = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-base/1")        
            else:
                model = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-large/1")            
            
            print( model.summary() )
            # model(english_sentences)

        #nprocs=mp.cpu_count() 
        #print( nprocs, ' processes' )
        nprocs = 2
        
        if 0:
            with Pool( nprocs ) as pool:                   
                Embeddings[emb] = list( tqdm( pool.imap( embed, inp ) ))
        else:            
            Embeddings[emb]  = embed(inp)
        
        exec_time = (time() - starttime)/60 
                
        d=Embeddings[emb].shape[1]
        print(d,'dimensions')
        
        pref = f'{src}_n{size_n}_emb{emb}_d{d}_{today}'
        f=open( pref+'.txt' ,'w')
        f.write( 'Finished in %2.f minutes'%exec_time )
        f.close()
        
        with open( pref+'".pkl', 'wb') as handle:
            pickle.dump( {"embeddings": Embeddings[emb] } , handle)   
        print('src', emb, 'done in ', exec_time, 'written to', pref )         
                
 
