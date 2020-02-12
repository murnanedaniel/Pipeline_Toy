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

* **Move all functions into prepare.py, train.py and evaluate.py**
* **Remove model file (otherwise `dvc run` fails!)**
* **git ac**

For the moment, only the train file is really doing anything interesting. Let us run the same training as in the Jupyter notebook, but now we will run it from the command-line, wrapped in a `dvc run` command.
```
dvc run -f train.dvc \ 
        -d src/train.py \
        -d data/ \
        -o cifar_net.pth \
        python src/train.py data/ cifar_net.pth
```
Essentially, this is building a pipeline that DVC can keep track of - consisting of "dependencies" and "staging" files. The tag `-f` allows us to name this particular "stage", and the `-d` tags tell DVC what this stage depends on. Any output goes to a `-o` tag. This is all followed by the actual commands of the stage, in this case a python execution. This can be pictured as
```
dependencies  ----> stage ----> outputs
```

* **git ac and dvc push**
* **Check that the pipeline exists with a visualisation**

We see:

```
+----------+   
| data.dvc |   
+----------+   
      *        
      *        
      *        
+-----------+  
| train.dvc |  
+-----------+ 
```

* **Add evaluation to the pipeline**
* **Add metric**

We will add an evaluation step to get some information about how the model performs. To start, let's just save the accuracy, and the time it took to run the inference of the model. These are two of the most important factors in deciding which model we will eventually ship.

```
dvc run -f evaluate.dvc \
        -d src/evaluate.py \
        -d cifar_net.pth \
        -d data \
        -m acc.metric \
        python src/evaluate.py data cifar_net.pth acc.metric
```

* **Check the full pipeline with visualisation**

Let's make sure that running this stage has integrated into the pipeline, with `dvc pipeline show --ascii evaluate.dvc`, and we get

```
  +----------+   
  | data.dvc |   
  +----------+   
        *        
        *        
        *        
  +-----------+  
  | train.dvc |  
  +-----------+  
        *        
        *        
        *        
+--------------+ 
| evaluate.dvc | 
+--------------+
```
Whereas, running `dvc pipeline show --ascii evaluate.dvc --outs` gives the data dependencies
```

                            +--------------+            +------+  
                            | src/train.py |          **| data |  
                            +--------------+       ***  +------+  
                                    *         *****         *     
                                    *      ***              *     
                                    *   ***                 *     
+-----------------+        +---------------+              ***     
| src/evaluate.py |        | cifar_net.pth |          ****        
+-----------------+**      +---------------+      ****            
                     ****          *          ****                
                         ****      *      ****                    
                             ***   *   ***                        
                            +------------+                        
                            | acc.metric |                        
                            +------------+  
```

This means: the input `data` and `train.py` files are independent, the model file depends on the network and data, and the final metrics depends on the model file and the data its evaluated on.

Try running `dvc repro evaluate.dvc`, nothing should happen since all the data and model files are in sync with the outputs and metrics.

Let's establish this first run as the baseline to improve upon. We'll commit the run 
```
git ac -m "Evaluate baseline"
dvc push
```
and tag it
```
git tag -a "baseline" -m "Evaluate baseline"
```
to give some intuition to when we compare back to it. Note that if we then push the commit, we need to add a `tag` flag
```
git push origin tag baseline
``` 
or `git push --tags` to push all tags. 

* **Tag this run as baseline**

### Changing the Model

* Update some hyperparameters
Let's make the model wider (i.e. increase the size of the hidden layer of the convolution). First, refactor out the model information into a file called `convNet.py` in a `src/models/` directory. We will import this model into the evaluation and training with 

``` from models.convNet import Net```

Now increase the second and first arguments of the first and second convolution layers (respectively) to something like 36. Now the beauty of DVC: simply run 

```
dvc repro evaluate.dvc
```

and all of the steps leading to the new metric evaluation will be completed, with dependencies correctly handled. Go ahead and comm

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



