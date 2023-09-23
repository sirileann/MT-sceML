# Spinal Curvature Estimation using Machine Learning (sceML)

sceML is the spinal curvature estimation using machine learning master project from Siri Rüegg (sirrueeg@ethz.ch)

## Get started:

### Installation Guide:

1. Clone msc-sirirueegg-sceML repository 

```bash 
git clone git@gitlab.ethz.ch:4d-spine/msc-sirirueegg-sceML.git 
``` 

2. Create virtual environment and activate it.  

```bash 
# Venv 
cd msc-sirirueegg-sceML 
python3.8 -m venv venv 
source venv/bin/activate 

# Conda
cd msc-sirirueegg-sceML 
conda create -n venv pip
conda activate venv
``` 

3. Install requirements
```bash
pip install -r requirements.txt 
``` 

## Data Pre-Processing

Pipelines to preprocess and create datasets are pre-implemented in *data_loader.py* and can be uncommented. 
The following pipelines are available:
- preprocess and create new (industry-referenced) dataset from raw data
- clean invalid scans from italian dataset
- preprocess and create new (ground-truth) dataset from raw data

```bash
cd data_preprocessing
```

**NOTE:** The pre-implemented pipelines are highly dependent on the data structure and especially on the naming conventions of the file-names and the json-keys. In case the naming conventions are different it is recomended to check the configurations in the functions *get_and_parse_data* or *create_dataset_italian* in *data_loader.py* and adjust them accordingly.

## Deep Learning Modules

### CNN's (2D Approach)

1. Train
```bash
cd ConvNets
python convNet_training.py fit --config convNet_config_train.yaml
```
2. Test
```bash
cd ConvNets
python convNet_testing.py test --config convNet_config_test.yaml
```

### Point Cloud Transformer (3D Approach)

Installation Instructions:
```bash
cd PointCloudTransformer
pip install pointnet2_ops_lib/.
```

1. Train
```bash
cd PointCloudTransformer
python pct_training.py fit --config pct_config_train.yaml
```
2. Test
```bash
cd PointCloudTransformer
python pct_testing.py test --config pct_config_test.yaml
```

#### Implemented Data Modules
Several modules are pre-implemented and can be used by configuring the *pct_config_train.yaml* accordingly. 

- **Back:** model.submodel = 'pct', data.class_path = 'src.pct_dataModule.Back_DataModule'
- **FixPoints:** model.submodel = 'fixPoints', data.class_path = 'src.pct_dataModule.FixPoint_DataModule'
- **Back+Depthmap:** model.submodel = 'backPC_backDM', data.class_path = 'src.pct_dataModule.BackPC_BackDM_DataModule'
- **Back+FixPoints:** model.submodel = 'back_fix', data.class_path = 'src.pct_dataModule.Back_Fix_DataModule'
- **Back+ESL+FixPoints:** model.submodel = 'back_esl_fix', data.class_path = 'src.pct_dataModule.Back_ESL_Fix_DataModule'

