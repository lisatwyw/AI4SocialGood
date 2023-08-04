

'''
X = [tokens+([0]* (max_tokens-len(tokens))) if len(tokens)<max_tokens else tokens[:max_tokens] for tokens in X] ## Bringing all samples to max_tokens length.
'''

from tensorflow.keras import backend as K


### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)
def f1macro(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    nclasses =len( np.unique( y_true ))
    precision=np.zeros(nclasses)
    recall=np.zeros(nclasses)
    f=0
    for c in range(nclasses):
        y=(y_true== c)*1.
        yp=(y_pred ==c)*1.
        precision[c], recall[c] = precision_m(y, yp), recall_m(y, yp)
        f += 2*((precision[c]*recall[c])/(precision[c]+recall[c]+K.epsilon()))
    return f/nclasses
