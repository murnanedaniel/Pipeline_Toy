# Pipelines with DVC and W&B

## Prelim

* Install everything!

To begin with, we need to install DVC and Weights and Biases.
(Use virtual environment?)
```
pip install --user dvc wandb
```

For Cori, module load pytorch/v1.2.0-gpu

## Tutorial

We will proceed with the tutorial in a way that we would actually explore data, and experiment with DNN models. To begin, we will fire up a Jupyter notebook to ensure that our data makes sense, and we can train at least one model on it.

### Jupyter Exploration

* **Import everything**
* **Download cifar with torch.datasets**
* **git / dvc init**
* **dvc add data/**
* **Add remote**
* **git remote**
* **Run some training tests**

### Staging Files

Now we have a sanity check on the data and model, we should move to a pipeline so that we are not glued to Jupyter. We abstract out the general sections of data preparation, model training, and evaluation into three files
  
    prepare.py, train.py, evaluate.py
    
With minimal changes, we should be able to pull the relevant Jupyter sections into these files. See the completed files for tips on doing this. 

* Move all functions into prepare.py, train.py and evaluate.py
* Remove model file (otherwise `dvc run` fails!)
* git add and dvc push

For the moment, only the train file is really doing anything interesting. Let us run the same training as in the Jupyter notebook, but now we will run it from the command-line, wrapped in a `dvc run` command.
```
dvc run ...
```
Essentially, this is building a pipeline that DVC can keep track of - consisting of "dependencies" and "staging" files. The tag `-f` allows us to name this particular "stage", and the `-d` tags tell DVC what this stage depends on. Any output goes to a `-o` tag. This is all followed by the actual commands of the stage, in this case a python execution. This can be pictured as
```
dependencies  ----> stage ----> outputs
```

* Check that the pipeline exists with a visualisation
* Add evaluation to the pipeline
* Add metric
* Check the full pipeline with visualisation
* Tag this run as baseline

### Changing the Model

* Update some hyperparameters
* git commit, dvc push
* Run the model with `dvc repro`
* Checkout to go back to master
* Compare metrics

### Visualising Performance 

* Include W&B in training
* Show some plots


### Update the Dataset

* Add some transformations / augmentations to prepare.py
* Run this into the pipeline
* Show when this runs, and when it doesn't need to



