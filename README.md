# AI4SocialGood

## Kaggle-Colab
[How to use Kaggle datasets with your Colab](Kaggle2Colab.ipynb)

## Ideas

```
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso #Loading the dataset
reg = LassoCV()
reg.fit(X, y)
```


[BERT-seq](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9028677/)


## Packaging models

```
tflite_model = keras_model_converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
with open('inference_args.json', "w") as f:
    json.dump({"selected_columns" : SEL_COLS}, f)
    
!zip submission.zip  './model.tflite' './inference_args.json'
```
