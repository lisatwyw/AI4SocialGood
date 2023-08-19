import gc
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf



with open ("../input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)


def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)


df = pd.read_csv('../input/asl-fingerspelling/train.csv')

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE] + [f'x_face_{i}' for i in LIP]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE] + [f'y_face_{i}' for i in LIP]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE] + [f'z_face_{i}' for i in LIP]

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

file_id = df.file_id.iloc[0]
inpdir = "../input/asl-fingerspelling/train_landmarks"
pqfile = f"{inpdir}/{file_id}.parquet"
seq_refs = df.loc[df.file_id == file_id]
seqs = load_relevant_data_subset(pqfile)

seq_id = seq_refs.sequence_id.iloc[0]
frames = seqs.iloc[seqs.index == seq_id].to_numpy()
phrase = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])


def process(x):
    lip_x = x[:, LIP_IDX_X]
    lip_y = x[:, LIP_IDX_Y]
    lip_z = x[:, LIP_IDX_Z]

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
    
    lip = np.stack([lip_x, lip_y, lip_z], axis=-1)[lpnonans]

    return rhand, lhand, rpose, lpose, lip

rhand, lhand, rpose, lpose, lip = process(frames)
print(rhand.shape, lhand.shape, rpose.shape, lpose.shape, lip.shape)
def gen(df):
    for file_id in df.file_id.unique():
        pqfile = f"../input/asl-fingerspelling/train_landmarks/{file_id}.parquet"
        seq_refs = df.loc[df.file_id == file_id]
        seqs = load_relevant_data_subset(pqfile)

        for seq_id in seq_refs.sequence_id:
            x = seqs.iloc[seqs.index == seq_id].to_numpy()
            y = df.loc[df.sequence_id == seq_id].phrase.iloc[0]
            rhand, lhand, rpose, lpose, lip = process(x)
            
            if max(rhand.shape[0], lhand.shape[0]) > len(y):
                yield rhand, lhand, rpose, lpose, lip
RHAND = []
LHAND = []
RPOSE = []
LPOSE = []
LIP = []

for rhand, lhand, rpose, lpose, lip in tqdm(gen(df)):
    RHAND.extend(rhand)
    LHAND.extend(lhand)
    RPOSE.extend(rpose)
    LPOSE.extend(lpose)
    LIP.extend(lip)
    
RHAND = np.array(RHAND)
LHAND = np.array(LHAND)
RPOSE = np.array(RPOSE)
LPOSE = np.array(LPOSE)
LIP = np.array(LIP)
gc.collect()

rh_mean = np.mean(RHAND, axis=0)
lh_mean = np.mean(LHAND, axis=0)
rp_mean = np.mean(RPOSE, axis=0)
lp_mean = np.mean(LPOSE, axis=0)
lip_mean = np.mean(LIP, axis=0)

rh_std = np.std(RHAND, axis=0)
lh_std = np.std(LHAND, axis=0)
rp_std = np.std(RPOSE, axis=0)
lp_std = np.std(LPOSE, axis=0)
lip_std = np.std(LIP, axis=0)

!mkdir stats
np.save("stats/rh_mean.npy", rh_mean)
np.save("stats/lh_mean.npy", lh_mean)
np.save("stats/rp_mean.npy", rp_mean)
np.save("stats/lp_mean.npy", lp_mean)
np.save("stats/lip_mean.npy", lip_mean)

np.save("stats/rh_std.npy", rh_std)
np.save("stats/lh_std.npy", lh_std)
np.save("stats/rp_std.npy", rp_std)
np.save("stats/lp_std.npy", lp_std)
np.save("stats/lip_std.npy", lip_std)


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
    
    lip   = tf.concat([lip_x[..., tf.newaxis], lip_y[..., tf.newaxis], lip_z[..., tf.newaxis]], axis=-1)
    rhand = tf.concat([rhand_x[..., tf.newaxis], rhand_y[..., tf.newaxis], rhand_z[..., tf.newaxis]], axis=-1)
    lhand = tf.concat([lhand_x[..., tf.newaxis], lhand_y[..., tf.newaxis], lhand_z[..., tf.newaxis]], axis=-1)
    rpose = tf.concat([rpose_x[..., tf.newaxis], rpose_y[..., tf.newaxis], rpose_z[..., tf.newaxis]], axis=-1)
    lpose = tf.concat([lpose_x[..., tf.newaxis], lpose_y[..., tf.newaxis], lpose_z[..., tf.newaxis]], axis=-1)
    
    hand = tf.concat([rhand, lhand], axis=1)
    hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
    mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)

    lip = lip[mask]
    rhand = rhand[mask]
    lhand = lhand[mask]
    rpose = rpose[mask]
    lpose = lpose[mask]

    return lip, rhand, lhand, rpose, lpose

pre_process0(frames)
def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

if not os.path.isdir("tfds"): os.mkdir("tfds")

for file_id in tqdm(df.file_id.unique()):
    pqfile = f"{inpdir}/{file_id}.parquet"
    tffile = f"tfds/{file_id}.tfrecord"
    seq_refs = df.loc[df.file_id == file_id]
    seqs = load_relevant_data_subset(pqfile)
    
    with tf.io.TFRecordWriter(tffile) as file_writer:
        for seq_id, phrase in zip(seq_refs.sequence_id, seq_refs.phrase):
            frames = seqs.iloc[seqs.index == seq_id].to_numpy()
            
            lip, rhand, lhand, rpose, lpose = pre_process0(frames)

            if max(rhand.shape[0], lhand.shape[0]) < len(phrase):
                continue
                
            features = {}
            features["lip"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lip, -1).numpy())) 
            features["rhand"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(rhand, -1).numpy())) 
            features["lhand"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lhand, -1).numpy())) 
            features["rpose"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(rpose, -1).numpy())) 
            features["lpose"] = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(lpose, -1).numpy())) 
            features["phrase"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[char_to_num[x] for x in phrase]))
            
            record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
            file_writer.write(record_bytes)


def decode_fn(record_bytes):
    schema = {
        "lip": tf.io.VarLenFeature(tf.float32),
        "rhand": tf.io.VarLenFeature(tf.float32),
        "lhand": tf.io.VarLenFeature(tf.float32),
        "rpose": tf.io.VarLenFeature(tf.float32),
        "lpose": tf.io.VarLenFeature(tf.float32),
        "phrase": tf.io.VarLenFeature(tf.int64)
    }
    x = tf.io.parse_single_example(record_bytes, schema)

    lip = tf.reshape(tf.sparse.to_dense(x["lip"]), (-1, 40, 3))
    rhand = tf.reshape(tf.sparse.to_dense(x["rhand"]), (-1, 21, 3))
    lhand = tf.reshape(tf.sparse.to_dense(x["lhand"]), (-1, 21, 3))
    rpose = tf.reshape(tf.sparse.to_dense(x["rpose"]), (-1, 5, 3))
    lpose = tf.reshape(tf.sparse.to_dense(x["lpose"]), (-1, 5, 3))
    phrase = tf.sparse.to_dense(x["phrase"])

    return lip, rhand, lhand, rpose, lpose, phrase

    
tffiles = [f"tfds/{file_id}.tfrecord" for file_id in df.file_id.unique()]
for batch in tf.data.TFRecordDataset(tffiles).map(decode_fn).take(1):
    print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape, batch[5])
