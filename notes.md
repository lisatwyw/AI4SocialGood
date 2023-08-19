## ```Polars```
```
import polars as pl

rolling_stones = """
linenum,last_name,first_name
1,Jagger,Mick
2,O"Brian,Mary
3,Richards,Keith
4,L"Etoile,Bennet
5,Watts,Charlie
6,Smith,D"Shawn
7,Wyman,Bill
8,Woods,Ron
9,Jones,Brian
"""

print(pl.read_csv(rolling_stones.encode(), quote_char=None))
```

## ```tfa```

[tfa-equivalence](https://docs.google.com/spreadsheets/d/1YMPudb7Otqx_TQu_oTHMm5IRZolyPcah8-qkMs_wI-I/)

## Kaggle-Colab
[How to use Kaggle datasets with your Colab](Kaggle2Colab.ipynb)

```
/opt/conda/lib/python3.10/site-packages 
```

## Number of model parameters in Pytorch and Tensorflow
https://wandb.ai/wandb_fc/tips/reports/How-to-Calculate-Number-of-Model-Parameters-for-PyTorch-and-Tensorflow-Models--VmlldzoyMDYyNzIx



## Packaging models

```
tflitemodel_base = TFLiteModel(model)
tflite_model = keras_model_converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
with open('inference_args.json', "w") as f:
    json.dump({"selected_columns" : SEL_COLS}, f)
    
!zip submission.zip  './model.tflite' './inference_args.json'
```

## Ideas

```
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso #Loading the dataset
reg = LassoCV()
reg.fit(X, y)
```
## XAI

https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_Introduction_to_AI_Interpretability_Course_2022.ipynb

## Misc

https://huggingface.co/spaces/xdecoder/SEEM
[BERT-seq](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9028677/)



