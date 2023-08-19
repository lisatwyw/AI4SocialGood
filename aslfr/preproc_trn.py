import gc,os,math
import pickle,json
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

IS_INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'

df = pd.read_csv('../input/asl-fingerspelling/train.csv')
meta_df = pd.read_csv('../input/asl-fingerspelling/supplemental_metadata.csv')
inpdir = "../input/asl-fingerspelling/train_landmarks"
meta_inpdir = '../input/asl-fingerspelling/supplemental_landmarks'

with open ("../input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)
    
num_to_char = {j:i for i,j in char_to_num.items()}
def num_to_char_fn(y):
    return [num_to_char.get(x, "") for x in y]



# ======================== hyperparams / admin vars
NORM_COORDS = 1
FRAME_LEN = 128

MAX_PHRASE_LENGTH = 64

NTSTS = 2
PREPROC = DEBUG = 0

N_ATT_HEADS = 12

trn_batch_size = 128
val_batch_size = 64

pad_token = '^'
pad_token_idx = 59

NBLOCKS=6
INCLUDE_EYEBROWS=1

N_EPOCHS = 2 if IS_INTERACTIVE else 70
N_WARMUP_EPOCHS = 0 if IS_INTERACTIVE else 10


LR_MAX = 1e-3
WD_RATIO = 0.05
WARMUP_METHOD = "exp"


prefix = f'../working/NFR{FRAME_LEN}_NBK{NBLOCKS}_NRM{NORM_COORDS}_NAH{N_ATT_HEADS}'
prefix


LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

L_EBROW = [383, 300, 293, 334, 296, 336, 285, 417]
L_EBROW += [265, 353, 276, 283, 282, 295]
R_EBROW = [156, 70, 63, 105, 66, 107, 55, 193]
R_EBROW += [35, 124, 46, 53, 52, 65]

X2 = [f'x_face_{i}' for i in L_EBROW] 
Y2 = [f'y_face_{i}' for i in L_EBROW] 
Z2 = [f'z_face_{i}' for i in L_EBROW]

X2+= [f'x_face_{i}' for i in R_EBROW] 
Y2+= [f'y_face_{i}' for i in R_EBROW] 
Z2+= [f'z_face_{i}' for i in R_EBROW]

SEL_COLS2 = X2 + Y2 + Z2

EBS_IDX_X   = [i for i, col in enumerate(SEL_COLS2)  if  "face" in col and "x" in col]
EBS_IDX_Y   = [i for i, col in enumerate(SEL_COLS2)  if  "face" in col and "y" in col]
EBS_IDX_Z   = [i for i, col in enumerate(SEL_COLS2)  if  "face" in col and "z" in col]

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] 
X+= [f'x_pose_{i}' for i in POSE] + [f'x_face_{i}' for i in LIP]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] 
Y+= [f'y_pose_{i}' for i in POSE] + [f'y_face_{i}' for i in LIP]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] 
Z+= [f'z_pose_{i}' for i in POSE] + [f'z_face_{i}' for i in LIP]

SEL_COLS = X + Y + Z

LIP_IDX_X   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "x" in col]
RHAND_IDX_X = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "x" in col]
LHAND_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "x" in col]

RPOSE_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "x" in col]
LPOSE_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "x" in col]

LIP_IDX_Y   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "y" in col]
RHAND_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "y" in col]
LHAND_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "y" in col]
RPOSE_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "y" in col]
LPOSE_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "y" in col]

LIP_IDX_Z   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "z" in col]
RHAND_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "z" in col]
LHAND_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "z" in col]
RPOSE_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "z" in col]
LPOSE_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "z" in col]

SEL_COLS += SEL_COLS2

## Define ```process*```, ```load*``` functions, etc.

def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

def process(x):

    rhand_x = x[:, RHAND_IDX_X]
    rhand_y = x[:, RHAND_IDX_Y]
    rhand_z = x[:, RHAND_IDX_Z]
    
    lhand_x = x[:, LHAND_IDX_X]
    lhand_y = x[:, LHAND_IDX_Y]
    lhand_z = x[:, LHAND_IDX_Z]

    rpose_x = x[:, RPOSE_IDX_X]
    rpose_y = x[:, RPOSE_IDX_Y]
    rpose_z = x[:, RPOSE_IDX_Z]
    
    lpose_x = x[:, LPOSE_IDX_X]
    lpose_y = x[:, LPOSE_IDX_Y]
    lpose_z = x[:, LPOSE_IDX_Z]
    
    rhnonans = ~np.isnan(np.sum(rhand_x, axis=1))
    lhnonans = ~np.isnan(np.sum(lhand_x, axis=1))
    lpnonans = ~np.isnan(np.sum(lip_x,   axis=1))
    
    rhand = np.stack([rhand_x, rhand_y, rhand_z], axis=-1)[rhnonans]
    rpose = np.stack([rpose_x, rpose_y, rpose_z], axis=-1)[rhnonans]
    
    lhand = np.stack([lhand_x, lhand_y, lhand_z], axis=-1)[lhnonans]
    lpose = np.stack([lpose_x, lpose_y, lpose_z], axis=-1)[lhnonans]

    lip_x = x[:, LIP_IDX_X]
    lip_y = x[:, LIP_IDX_Y]
    lip_z = x[:, LIP_IDX_Z]
    lip = np.stack([lip_x, lip_y, lip_z], axis=-1)[lpnonans]

    ebs_x = x[:, EBS_IDX_X]
    ebs_y = x[:, EBS_IDX_Y]
    ebs_z = x[:, EBS_IDX_Z]
    ebsnonans = ~np.isnan(np.sum(ebs_x,   axis=1))
    ebs = np.stack([ebs_x, ebs_y, ebs_z], axis=-1)[ebsnonans]

    return rhand, lhand, rpose, lpose, lip, ebs

def gen(df):
    for file_id in df.file_id.unique():
        pqfile = f"/kaggle/input/asl-fingerspelling/train_landmarks/{file_id}.parquet"
        seq_refs = df.loc[df.file_id == file_id]
        seqs = load_relevant_data_subset(pqfile)

        for seq_id in seq_refs.sequence_id:
            x = seqs.iloc[seqs.index == seq_id].to_numpy()
            y = df.loc[df.sequence_id == seq_id].phrase.iloc[0]
            rhand, lhand, rpose, lpose, lip, ebs = process(x)
            
            if max(rhand.shape[0], lhand.shape[0]) > len(y):
                yield rhand, lhand, rpose, lpose, lip, ebs
                
                
@tf.function(jit_compile=True)
def pre_process0(x):
    lip_x = tf.gather(x, LIP_IDX_X, axis=1)
    lip_y = tf.gather(x, LIP_IDX_Y, axis=1)
    lip_z = tf.gather(x, LIP_IDX_Z, axis=1)

    rhand_x = tf.gather(x, RHAND_IDX_X, axis=1)
    rhand_y = tf.gather(x, RHAND_IDX_Y, axis=1)
    rhand_z = tf.gather(x, RHAND_IDX_Z, axis=1)
    
    lhand_x = tf.gather(x, LHAND_IDX_X, axis=1)
    lhand_y = tf.gather(x, LHAND_IDX_Y, axis=1)
    lhand_z = tf.gather(x, LHAND_IDX_Z, axis=1)

    rpose_x = tf.gather(x, RPOSE_IDX_X, axis=1)
    rpose_y = tf.gather(x, RPOSE_IDX_Y, axis=1)
    rpose_z = tf.gather(x, RPOSE_IDX_Z, axis=1)
    
    lpose_x = tf.gather(x, LPOSE_IDX_X, axis=1)
    lpose_y = tf.gather(x, LPOSE_IDX_Y, axis=1)
    lpose_z = tf.gather(x, LPOSE_IDX_Z, axis=1)

    ebs_x = tf.gather(x, EBS_IDX_X, axis=1)
    ebs_y = tf.gather(x, EBS_IDX_Y, axis=1)
    ebs_z = tf.gather(x, EBS_IDX_Z, axis=1)
    
    lip   = tf.concat([lip_x[..., tf.newaxis], lip_y[..., tf.newaxis], lip_z[..., tf.newaxis]], axis=-1)
    rhand = tf.concat([rhand_x[..., tf.newaxis], rhand_y[..., tf.newaxis], rhand_z[..., tf.newaxis]], axis=-1)
    lhand = tf.concat([lhand_x[..., tf.newaxis], lhand_y[..., tf.newaxis], lhand_z[..., tf.newaxis]], axis=-1)
    rpose = tf.concat([rpose_x[..., tf.newaxis], rpose_y[..., tf.newaxis], rpose_z[..., tf.newaxis]], axis=-1)
    lpose = tf.concat([lpose_x[..., tf.newaxis], lpose_y[..., tf.newaxis], lpose_z[..., tf.newaxis]], axis=-1)
    ebs   = tf.concat([ebs_x[..., tf.newaxis], ebs_y[..., tf.newaxis], ebs_z[..., tf.newaxis]], axis=-1)
   
    hand = tf.concat([rhand, lhand], axis=1)
    hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
    
    mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)

    lip = lip[mask]
    rhand = rhand[mask]
    lhand = lhand[mask]
    rpose = rpose[mask]
    lpose = lpose[mask]
    ebs = ebs[mask]

    return lip, rhand, lhand, rpose, lpose, ebs


@tf.function()
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]), constant_values=float("NaN"))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x
 
if NORM_COORDS:
    # mean coords  # /kaggle/input/
    RHM = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/rh_mean.npy")
    LHM = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lh_mean.npy")
    RPM = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/rp_mean.npy")
    LPM = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lp_mean.npy")
    LIPM= np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lip_mean.npy")
    EBM = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/ebs_mean.npy")

    # STD coords
    RHS = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/rh_std.npy")
    LHS = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lh_std.npy")
    RPS = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/rp_std.npy")
    LPS = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lp_std.npy")
    LIPS= np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/lip_std.npy")
    EBS = np.load("/kaggle/input/aslfr-120landmarks-train-supp/mean_std/ebs_std.npy")

@tf.function()
def resize_pad_norm(lip, rhand, lhand, rpose, lpose, eyebrows  ): 
    lip   = resize_pad(lip) 
    rhand = resize_pad(rhand)  
    lhand = resize_pad(lhand) 
    rpose = resize_pad(rpose)  
    lpose = resize_pad(lpose)     
    eyebrows = resize_pad(eyebrows)

    if NORM_COORDS:
        lip   = ((lip) - LIPM) / LIPS
        rhand = ((rhand) - RHM) / RHS
        lhand = ((lhand) - LHM) / LHS
        rpose = ((rpose) - RPM) / RPS
        lpose = ((lpose) - LPM) / LPS
        eyebrows = ((eyebrows) - EBM) / EBS

    x = tf.concat([lip, rhand, lhand, rpose, lpose, eyebrows], axis=1) 
    s = tf.shape(x)
    x = tf.reshape(x, (s[0], s[1]*s[2]))
    x = tf.where(tf.math.is_nan(x), 0.0, x)
    return x


# test on 1
file_id = df.file_id.iloc[0]

pqfile = f"{inpdir}/{file_id}.parquet"
seq_refs = df.loc[df.file_id == file_id]
seqs = load_relevant_data_subset(pqfile)

seq_id = seq_refs.sequence_id.iloc[0]

pid = seq_refs.participant_id.iloc[0]
frames = seqs.iloc[seqs.index == seq_id].to_numpy()
#phrase = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])

rhand, lhand, rpose, lpose, lip, eyebrows = process(frames)

pre0 = pre_process0(frames)

if PREPROC==0:
    pre1 = resize_pad_norm(*pre0 )
    INPUT_SHAPE = list(pre1.shape)
    print('Size of input shape (n-frames, n-landmark-coords)',INPUT_SHAPE)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:32.888246Z","iopub.execute_input":"2023-08-16T15:31:32.891621Z","iopub.status.idle":"2023-08-16T15:31:32.904178Z","shell.execute_reply.started":"2023-08-16T15:31:32.891582Z","shell.execute_reply":"2023-08-16T15:31:32.902145Z"}}
df.participant_id.unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:32.905533Z","iopub.execute_input":"2023-08-16T15:31:32.906415Z","iopub.status.idle":"2023-08-16T15:31:32.920700Z","shell.execute_reply.started":"2023-08-16T15:31:32.906322Z","shell.execute_reply":"2023-08-16T15:31:32.919180Z"}}
meta_df.participant_id.unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:32.922902Z","iopub.execute_input":"2023-08-16T15:31:32.925319Z","iopub.status.idle":"2023-08-16T15:31:32.955772Z","shell.execute_reply.started":"2023-08-16T15:31:32.925284Z","shell.execute_reply":"2023-08-16T15:31:32.954719Z"}}
df2 =df.sort_values('participant_id')
df2.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:32.960329Z","iopub.execute_input":"2023-08-16T15:31:32.960971Z","iopub.status.idle":"2023-08-16T15:31:32.985885Z","shell.execute_reply.started":"2023-08-16T15:31:32.960938Z","shell.execute_reply":"2023-08-16T15:31:32.985093Z"}}
meta_df2 =meta_df.sort_values('participant_id')
meta_df2.head(2)

# %% [markdown]
# # Steps in ```PREPROC```

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:32.992546Z","iopub.execute_input":"2023-08-16T15:31:32.994702Z","iopub.status.idle":"2023-08-16T15:31:33.021002Z","shell.execute_reply.started":"2023-08-16T15:31:32.994669Z","shell.execute_reply":"2023-08-16T15:31:33.020218Z"}}
%%time

def rewrite(inpdir, df, suffix='', always=0, pids=None):

    if not os.path.isdir("tfds"+suffix): os.mkdir("tfds"+suffix)

    participants = {}

    for file_id in tqdm(df.file_id.unique()):
        pqfile = f"{inpdir}/{file_id}.parquet"
        tffile = f"tfds{suffix}/{file_id}.tfrecord"
        seq_refs = df.loc[df.file_id == file_id]
        seqs = load_relevant_data_subset(pqfile)

        participants[file_id]= seq_refs.participant_id.unique()
        with tf.io.TFRecordWriter(tffile) as file_writer:
            for seq_id, pid, phrase in zip(seq_refs.sequence_id, seq_refs.participant_id, seq_refs.phrase):
                 
                if (always + (np.sum( pids == pid )==0 ).astype(int)):
                    frames = seqs.iloc[seqs.index == seq_id].to_numpy()
                    nframes = len(frames)

                    lip, rhand, lhand, rpose, lpose, ebs = pre_process0(frames)

                    if max(rhand.shape[0], lhand.shape[0]) < len(phrase):
                        continue

                    features = {}
                    features["lip"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lip, -1).numpy())) 
                    features["rhand"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(rhand, -1).numpy())) 
                    features["lhand"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lhand, -1).numpy())) 
                    features["rpose"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(rpose, -1).numpy())) 
                    features["lpose"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lpose, -1).numpy())) 
                    features["eyebrows"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(eyebrows, -1).numpy())) 
                    
                    features["phrase"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[char_to_num[x] for x in phrase]))
                    features["pid"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[pid])) 
                    features["nframes"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[nframes])) 

                    record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                    file_writer.write(record_bytes)

    file = open(f'files2pids_{suffix}.pkl', 'wb')

    try:
        # dump information to that file
        pickle.dump( participants, file)

        # close the file
        file.close()
    except:
        pass 
    
    
if PREPROC:
    rewrite( inpdir, df2, suffix='', always=1 )


# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:33.024869Z","iopub.execute_input":"2023-08-16T15:31:33.027022Z","iopub.status.idle":"2023-08-16T15:31:33.038486Z","shell.execute_reply.started":"2023-08-16T15:31:33.026988Z","shell.execute_reply":"2023-08-16T15:31:33.037672Z"}}
%time
if PREPROC:
    not_in_train = np.setdiff1d(meta_df2.participant_id, df2.participant_id )
    in_train = np.intersect1d(meta_df2.participant_id, df2.participant_id )
    
    rewrite( meta_inpdir, meta_df2, suffix='supp', pids=not_in_train)
    rewrite( meta_inpdir, meta_df2, suffix='overlap', pids= in_train)

# %% [markdown]
# # Steps in ```PREPROC``` (collect stats)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:33.042826Z","iopub.execute_input":"2023-08-16T15:31:33.045436Z","iopub.status.idle":"2023-08-16T15:31:33.068778Z","shell.execute_reply.started":"2023-08-16T15:31:33.045401Z","shell.execute_reply":"2023-08-16T15:31:33.067973Z"}}
%%time
if PREPROC:
    RHAND = []
    LHAND = []
    RPOSE = []
    LPOSE = []
    LIP = []
    EBS = []
    
    for rhand, lhand, rpose, lpose, lip, ebs in tqdm(gen(df)):
        RHAND.extend(rhand)
        LHAND.extend(lhand)
        RPOSE.extend(rpose)
        LPOSE.extend(lpose)
        LIP.extend(lip)
        EBS.extend(EBS)
        
    RHAND = np.array(RHAND)
    LHAND = np.array(LHAND)
    RPOSE = np.array(RPOSE)
    LPOSE = np.array(LPOSE)
    LIP = np.array(LIP)
    EBS = np.array(EBS)
    gc.collect()

    rh_mean = np.mean(RHAND, axis=0)
    lh_mean = np.mean(LHAND, axis=0)
    rp_mean = np.mean(RPOSE, axis=0)
    lp_mean = np.mean(LPOSE, axis=0)
    lip_mean = np.mean(LIP, axis=0)
    ebs_mean = np.mean(EBS, axis=0)

    rh_std = np.std(RHAND, axis=0)
    lh_std = np.std(LHAND, axis=0)
    rp_std = np.std(RPOSE, axis=0)
    lp_std = np.std(LPOSE, axis=0)
    lip_std = np.std(LIP, axis=0)
    ebs_std = np.std(EBS, axis=0)

    !mkdir mean_std
    np.save("mean_std/rh_mean.npy", rh_mean)
    np.save("mean_std/lh_mean.npy", lh_mean)
    np.save("mean_std/rp_mean.npy", rp_mean)
    np.save("mean_std/lp_mean.npy", lp_mean)
    np.save("mean_std/lip_mean.npy", lip_mean)
    np.save("mean_std/ebs_mean.npy", ebs_mean)

    np.save("mean_std/rh_std.npy", rh_std)
    np.save("mean_std/lh_std.npy", lh_std)
    np.save("mean_std/rp_std.npy", rp_std)
    np.save("mean_std/lp_std.npy", lp_std)
    np.save("mean_std/lip_std.npy", lip_std)       
    np.save("mean_std/ebs_std.npy", ebs_std)

# %% [markdown]
# # Define train/ val sets
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:33.072712Z","iopub.execute_input":"2023-08-16T15:31:33.074928Z","iopub.status.idle":"2023-08-16T15:31:39.717978Z","shell.execute_reply.started":"2023-08-16T15:31:33.074894Z","shell.execute_reply":"2023-08-16T15:31:39.716944Z"}}
def decode_fn(record_bytes):
    schema = {
        "lip": tf.io.VarLenFeature(tf.float32),
        "rhand": tf.io.VarLenFeature(tf.float32),
        "lhand": tf.io.VarLenFeature(tf.float32),
        "rpose": tf.io.VarLenFeature(tf.float32),
        "lpose": tf.io.VarLenFeature(tf.float32),
        "eyebrows": tf.io.VarLenFeature(tf.float32),
        "phrase": tf.io.VarLenFeature(tf.int64),
        "pid": tf.io.FixedLenFeature(1,tf.int64),
        "nframes": tf.io.FixedLenFeature(1,tf.int64)
    }
    x = tf.io.parse_single_example(record_bytes, schema)

    lip = tf.reshape(tf.sparse.to_dense(x["lip"]), (-1, 40, 3))
    rhand = tf.reshape(tf.sparse.to_dense(x["rhand"]), (-1, 21, 3))
    lhand = tf.reshape(tf.sparse.to_dense(x["lhand"]), (-1, 21, 3))
    rpose = tf.reshape(tf.sparse.to_dense(x["rpose"]), (-1, 5, 3))
    lpose = tf.reshape(tf.sparse.to_dense(x["lpose"]), (-1, 5, 3))
    eyebrows = tf.reshape(tf.sparse.to_dense(x["eyebrows"]), (-1, 28, 3))
    
    phrase = tf.sparse.to_dense(x["phrase"])
    pid = x["pid"]
    nframes = x["nframes"]

    return         lip, rhand, lhand, rpose, lpose, eyebrows, phrase, pid, nframes

def pre_process_fn(lip, rhand, lhand, rpose, lpose, eyebrows, phrase, pid, nframes):
    phrase = tf.pad(phrase, [[0, MAX_PHRASE_LENGTH-tf.shape(phrase)[0]]], constant_values=pad_token_idx)
    x = resize_pad_norm(lip, rhand, lhand, rpose, lpose, eyebrows ) 
    return x, phrase, (pid, nframes)

#  /kaggle/input//tfds
from glob import glob
RD,NVALS=1,20
if IS_INTERACTIVE:
    RD,NVALS=10,1
t_tffiles = glob( '../input/aslfr-120landmarks-train-supp/tfds/*tfrecord')[::RD]
print( f'{len(t_tffiles)} train files' ) 

if IS_INTERACTIVE==0:
    t_tffiles += glob('../input/aslfr-120landmarks-train-supp/tfdsoverlap/*tfrecord') 
    print( f'{len(t_tffiles)} train files from supplementary' ) 

v_tffiles = glob('../input/aslfr-120landmarks-train-supp/tfdssupp/*tfrecord')
print( f'{len(t_tffiles)} files for validation' ) 
       

    
trn_dataset =  tf.data.TFRecordDataset(t_tffiles[:-NTSTS])\
.prefetch(tf.data.AUTOTUNE).shuffle(5000).map(
    decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(
    pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(trn_batch_size).prefetch(
    tf.data.AUTOTUNE)
'''
tst_dataset =  tf.data.TFRecordDataset(t_tffiles[-NTSTS:])\
.prefetch(tf.data.AUTOTUNE).shuffle(5000).map(
    decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(
    pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(val_batch_size).prefetch(
    tf.data.AUTOTUNE)
'''

val_dataset =  tf.data.TFRecordDataset(v_tffiles).\
prefetch(tf.data.AUTOTUNE).map(
    decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(
    pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(val_batch_size).prefetch(
    tf.data.AUTOTUNE)


batch = next(iter(trn_dataset))
batch = next(iter(val_dataset))
#batch = next(iter(tst_dataset))
batch[0].shape, batch[1].shape

# %% [markdown]
# # Define model

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:31:39.719705Z","iopub.execute_input":"2023-08-16T15:31:39.720673Z","iopub.status.idle":"2023-08-16T15:32:24.371656Z","shell.execute_reply.started":"2023-08-16T15:31:39.720633Z","shell.execute_reply":"2023-08-16T15:32:24.370529Z"}}
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.4,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
        
        
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x

    
def TCNBlock(filters, kernel_size, dilation_rate, drop_rate=0.0, activation='relu', name=None):
    """
    Temporal Convolutional Block with Bahdanau Attention.
    """
    if name is None:
        name = str(tf.keras.backend.get_uid("tcnblock"))
    
    attention_layer = BahdanauAttention(filters)
    
    def apply(inputs):
        skip = inputs

        # 1. Causal Convolution
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, name=name + '_dwconv')(inputs)
        
        # 2. Batch Normalization and Activation
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn1')(x)
        x = tf.keras.layers.Activation(activation, name=name + '_act1')(x)

        # 3. Efficient Channel Attention
        x = ECA()(x)

        # 4. Apply Bahdanau Attention
        query = tf.keras.layers.Dense(filters)(x)
        x, attention_weights = attention_layer(query, x)

        # 5. Pointwise Convolution
        x = tf.keras.layers.Conv1D(filters, 1, activation=activation, name=name + '_conv1')(x)
        
        # 6. Another round of Batch Normalization
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn2')(x)
        
        # 7. Dropout for regularization
        x = tf.keras.layers.Dropout(drop_rate, name=name + '_drop')(x)
        
        # 8. Project input to the right shape for residual connection
        if tf.keras.backend.int_shape(skip)[-1] != filters:
            skip = tf.keras.layers.Conv1D(filters, 1, name=name + '_residual_proj')(skip)
            
        # 9. Residual connection
        x = tf.keras.layers.add([x, skip], name=name + '_add')
        
        return x

    return apply


def TransformerBlock(dim=256, num_heads=6, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def positional_encoding(maxlen, num_hid):
    depth = num_hid/2
    positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
    depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
    angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
    angle_rads = tf.linalg.matmul(positions, angle_rates)
    pos_encoding = tf.concat(
      [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
      axis=-1)
    return pos_encoding

def CTCLoss(labels, logits):
    label_length = tf.reduce_sum(tf.cast(labels != pad_token_idx, tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=pad_token_idx,
            logits_time_major=False
        )
    loss = tf.reduce_mean(loss)
    return loss

def get_model(dim = 384, num_blocks = 6, N_ATT_HEADS = 6, drop_rate  = 0.4):
    inp = tf.keras.Input(INPUT_SHAPE)
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    pe = tf.cast(positional_encoding(INPUT_SHAPE[0], dim), dtype=x.dtype)
    x = x + pe
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)
    
    for i in range(num_blocks):
        x = Conv1DBlock(dim, 11, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, 5, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, 3, drop_rate=drop_rate)(x)
        x = TransformerBlock(dim, expand=2, num_heads = N_ATT_HEADS )(x)

    x = tf.keras.layers.Dense(dim*2,activation='relu',name='top_conv')(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tf.keras.layers.Dense(len(char_to_num),name='classifier')(x)

    model = tf.keras.Model(inp, x)

    loss = CTCLoss
    
    # Adam Optimizer
    optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
    optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    model.compile(loss=loss, optimizer=optimizer)

    return model

tf.keras.backend.clear_session()
model = get_model( N_ATT_HEADS = N_ATT_HEADS )

from keras.utils.layer_utils import count_params

n_trainables = sum(count_params(layer) for layer in model.trainable_weights)
n_nontrainables = sum(count_params(layer) for layer in model.non_trainable_weights)

if IS_INTERACTIVE:
    #len(model.variables), len(model.trainable_variables), 
    print( n_trainables, 'trainables', n_nontrainables, 'non-trainable params.' )

# 8 attention heads: 18,320,790, 28,416 non-trainable
# 6 attention heads: 18,311,574, 28,416, non-trainable



# %% [markdown]
# # Test init submission package size

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:32:24.373565Z","iopub.execute_input":"2023-08-16T15:32:24.374323Z","iopub.status.idle":"2023-08-16T15:33:49.233750Z","shell.execute_reply.started":"2023-08-16T15:32:24.374284Z","shell.execute_reply":"2023-08-16T15:33:49.232250Z"}}
@tf.function()
def decode_phrase(pred):
    x = tf.argmax(pred, axis=1)
    diff = tf.not_equal(x[:-1], x[1:])
    adjacent_indices = tf.where(diff)[:, 0]
    x = tf.gather(x, adjacent_indices)
    mask = x != pad_token_idx
    x = tf.boolean_mask(x, mask, axis=0)
    return x

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text

class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(SEL_COLS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(SEL_COLS))), lambda: tf.identity(x))
        x = x[0]
        
        x = pre_process0(x)
        x = resize_pad_norm(*x)
        x = tf.reshape(x, INPUT_SHAPE)
        x = x[None]
        x = self.model(x, training=False)
        x = x[0]
        x = decode_phrase(x)
        x = tf.cond(tf.shape(x)[0] == 0, lambda: tf.zeros(1, tf.int64), lambda: tf.identity(x))
        x = tf.one_hot(x, 59)
        return {'outputs': x}

tflitemodel_base = TFLiteModel(model)
tflitemodel_base(frames)["outputs"].shape

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)

keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
keras_model_converter.target_spec.supported_types = [tf.float16]
tflite_model = keras_model_converter.convert()

if 0:
    with open('model0.tflite', 'wb') as f:
        f.write(tflite_model)

    with open('inference_args.json', "w") as f:
        json.dump({"selected_columns" : SEL_COLS}, f)

    #!zip submission0.zip  './model0.tflite' './inference_args.json'

    !ls -lstrh *zip

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:33:49.236218Z","iopub.execute_input":"2023-08-16T15:33:49.236660Z","iopub.status.idle":"2023-08-16T15:33:49.244512Z","shell.execute_reply.started":"2023-08-16T15:33:49.236614Z","shell.execute_reply":"2023-08-16T15:33:49.243467Z"}}


# %% [markdown]
# # Define callbacks

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:33:49.246421Z","iopub.execute_input":"2023-08-16T15:33:49.247144Z","iopub.status.idle":"2023-08-16T15:33:49.816455Z","shell.execute_reply.started":"2023-08-16T15:33:49.247110Z","shell.execute_reply":"2023-08-16T15:33:49.815443Z"}}


def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
    
def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1
    
    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])
    
    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)
    
    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)
    
    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()


# A callback class to output a few transcriptions during training
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        #model.save_weights("model.h5")
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y, meta = batch
            batch_predictions = model(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = "".join(num_to_char_fn(label.numpy()))
                targets.append(label)
        print("-" * 100)
        # for i in np.random.randint(0, len(predictions), 2):
        for i in range(32):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}, len: {len(predictions[i])}")
            print("-" * 100)


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')
        
# Learning rate for encoder
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]
# Plot Learning Rate Schedule
plot_lr_schedule(LR_SCHEDULE, epochs=N_EPOCHS)
# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)



#train_dataset =  tf.data.TFRecordDataset(tffiles[val_len:]).prefetch(tf.data.AUTOTUNE).shuffle(5000).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(train_batch_size).prefetch(tf.data.AUTOTUNE)
#val_dataset =  tf.data.TFRecordDataset(tffiles[:val_len]).prefetch(tf.data.AUTOTUNE).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(val_batch_size).prefetch(tf.data.AUTOTUNE)

# Callback function to check transcription on the val set.
validation_callback = CallbackEval(val_dataset.take(NVALS))

batch = next(iter(val_dataset))
batch[0].shape, batch[1].shape

# save every epoch
saver = tf.keras.callbacks.ModelCheckpoint(
    prefix + '_{epoch:03d}.h5',
    monitor= 'val_loss',
    verbose = 0,
    save_best_only= False,
    save_weights_only = True,
    mode = 'auto',
    save_freq='epoch',
    period=2,
    options=None,
    initial_value_threshold=None,
)



# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:33:49.817821Z","iopub.execute_input":"2023-08-16T15:33:49.818192Z","iopub.status.idle":"2023-08-16T15:39:24.479995Z","shell.execute_reply.started":"2023-08-16T15:33:49.818155Z","shell.execute_reply":"2023-08-16T15:39:24.478982Z"}}
%%time

if 1:
    history = model.fit(
    trn_dataset,
    validation_data=val_dataset,
    epochs=N_EPOCHS,
    callbacks=[
        validation_callback,
        lr_callback,
        saver,
        WeightDecayCallback(),
    ]
    )
    hist=history.history
    for k in hist.keys():
        plt.plot( hist[k], label=k )
    plt.legend()
    
    
    file = open('training_history.pkl', 'wb')
    pickle.dump(hist, file)
    file.close()

# %% [code]
 

# %% [markdown]
# # Submission

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T15:39:24.481885Z","iopub.execute_input":"2023-08-16T15:39:24.482282Z","iopub.status.idle":"2023-08-16T15:39:47.652112Z","shell.execute_reply.started":"2023-08-16T15:39:24.482242Z","shell.execute_reply":"2023-08-16T15:39:47.651104Z"}}
%%time

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)

keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
keras_model_converter.target_spec.supported_types = [tf.float16]

tflite_model = keras_model_converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
    
!zip submission.zip  './model.tflite' './inference_args.json'

