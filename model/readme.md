
### Files in the folder

- **data_generation.py** and **data_generation_icu.py**
	are the files that create smooth time-series representation from the options selected in **Block 7** of **mainPipeline.ipynb**.
  The output is saved in csv and dictionary format.
  
- **evaluation.py**
  contains code to perform evaluations on predictions made by model.
  It can also be used as standalone module whoch takes predictions and labels as input. 
  Refere to **Block 10** of **mainPipeline.ipynb** for details.
  
- **fairness.py**
  contains code to perform fairness evaluations on predictions made by model.
  It can also be used as standalone module whoch takes predictions, labels and list of sensitive attributes as input. 
  Refere to **Block 11** of **mainPipeline.ipynb** for details.
  
- **parameters.py**
  contains the list of hyperparameters for deep learning models.
  
- **ml_models.py**
  contains code to train and test machine learning models included in the pipeline.
  Refere to **Block 8** of **mainPipeline.ipynb** for details.
  
- **mimic_model.py**
  contains definition of deep learning models included in the pipeline.
  Refere to **Block 9** of **mainPipeline.ipynb** for details.
  
- **dl_train.py**
  contains the code to train and test deep learning models included in the pipeline.
  Refere to **Block 9** of **mainPipeline.ipynb** for details.


