import multiprocessing as mp
from multiprocessing import Pool    

import pandas as pd

from tqdm import tqdm 
from time import time
from datetime import date
today =date.today()

import pickle, json

try:
    import wget
except:
    exec( open('persist_install.py','r').read() )
    install_packages(['pip install wget'], INTERACTIVE )
    import wget

if ('decoded_df2' in globals())==False:
    print('reading decoded_df2...')
    
    decoded_df2=pd.read_csv('/kaggle/input/neiss-sentence-transform-embeddings/decoded_df2__l1.csv')
    decoded_df2=decoded_df2.drop_duplicates('cpsc_case_number',)    
    #decoded_df2.to_csv('decoded_df2_unique.csv')
    
    cpcs_nums = decoded_df2.cpsc_case_number
    size_n = decoded_df2.shape[0]
    
    Embeddings={}

def embed( model, sentences, TF_MODEL ):             
    if TF_MODEL==1:
        #embeddings = model.predict( sentences )
        embeddings = model(sentences)
    else: # pytorch         
        embeddings = model.encode(sentences)            
    return embeddings

def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds/norms

# ======== https://github.com/huggingface/transformers/issues/15038 
class NoDaemonProcess( mp.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type( mp.get_context("fork"))):
    Process = NoDaemonProcess

for src in [ 'narrative_cleaned']: #,'narrative' ]:    
    for emb in EMB:            
        if 1:            
            if INTERACTIVE:        
                inp = decoded_df2[src][:30] 
            else:
                inp =  decoded_df2[src]  
            if 'cleaned' not in src:        
                inp = inp.str.lower()
            inp = list(inp)

            starttime=time()           
            TF_MODEL = 0
            
            if emb<4: 
                try:
                    from sentence_transformers import SentenceTransformer
                except:                
                    install_packages( ['pip install -U sentence-transformers'], INTERACTIVE ); 
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

            elif emb==5:
                # [1] Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang. Language-agnostic BERT Sentence Embedding. July 2020
                import tensorflow_hub as hub            
                try:
                    import tensorflow_text as text  # Registers the ops.
                except:
                    install_packages(['pip install tensorflow_text'], INTERACTIVE)
                    import tensorflow_text as text  # Registers the ops.

                from transformers import pipeline 
                tokenizer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2')
                model = hub.KerasLayer('https://tfhub.dev/google/LaBSE/2')
                
                def compute( inp ):                                       
                    #print( end=f'{inp} ?', flush=True )
                    res= model( tokenizer( inp ) )                    
                    r=normalization( res['default']  )
                    return np.array(r)                                               
                mp_context = NoDaemonContext()                
                nprocs = mp.cpu_count()
                print( nprocs, 'cores' )
                with mp_context.Pool( nprocs ) as pool:
                    async_r = [pool.apply_async( compute, ii ).get() for i in tqdm(range(size_n)) for ii in inp[i] ] 
                    print( end='.' )
                
                '''
                def compute( inp ):
                    pip = get_pip() 
                pool = Pool( proprocess = 3 )
                preds = pool.map( compute, inp )
                pool.close()
                pool.join()                
                Embeddings[emb]= preds
                '''                
            elif emb>=6:

                import torch                                            
                from transformers import pipeline
                from torch.multiprocessing import Pool, Process, set_start_method
                #set_start_method("spawn", force=True)

                from transformers import BertModel, BertTokenizerFast         
                names ={}
                names[6] = "setu4993/LEALLA-small"
                names[7] = "setu4993/LEALLA-base"
                names[8] = "setu4993/LEALLA-large"                    

                def get_pipe():
                    tokenizer = BertTokenizerFast.from_pretrained(names[emb])
                    model =  BertModel.from_pretrained(names[emb])
                    model = model.eval()  
                    return tokenizer, model
                
                def compute( inp ):
                    tokenizer, model = get_pipe()                    
                    #print(len(inp), end=f' {inp}', )
                    english_inputs = tokenizer(inp, return_tensors="pt", padding=True)

                    with torch.no_grad():
                        english_outputs = model(**english_inputs)
                    r = np.array( english_outputs.pooler_output )    
                    return r
                
                multi_pool = Pool(processes=3)
                predictions = multi_pool.map( compute, inp)                 
                multi_pool.close()
                multi_pool.join()
                
                #m = predictions[0].shape[1] 
                #p = np.zeros( (size_n, m) )
                #predictions = [ p[i,:]=j for i,j in enumerate( predictions ) ]                
                Embeddings[emb]=np.array(predictions).squeeze()
                
        if emb<5:            
            Embeddings[emb] = embed( model, inp, TF_MODEL )            

        exec_time = (time() - starttime)/60 

        d=Embeddings[emb].shape[1]
        print(d,'dimensions')

        pref = f'{src}_n{size_n}_emb{emb}_d{d}_{today}'
        f=open( pref+'.txt' ,'w')
        f.write( 'Finished in %2.f minutes'%exec_time )
        f.close()

        with open( pref+'.pkl', 'wb') as handle:
            pickle.dump( {"embeddings": Embeddings[emb] } , handle)   
        print('src', emb, 'done in ', exec_time, 'written to', pref )                          
