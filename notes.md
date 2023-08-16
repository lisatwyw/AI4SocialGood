

## Kaggle-Colab
[How to use Kaggle datasets with your Colab](Kaggle2Colab.ipynb)

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
