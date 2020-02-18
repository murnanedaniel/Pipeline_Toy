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

We will proceed with the tutorial in a way that we would actually explore data, and experiment with DNN models. This means we will not take the fastest line to results, but meander and refactor to incorporate improvements to our pipeline along the way. To begin, we will fire up a Jupyter notebook to ensure that our data makes sense, and we can train at least one model on it.

### Jupyter Exploration

** Put the notebook in a /notebook subdirectory! **

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

and all of the steps leading to the new metric evaluation will be completed, with 
dependencies correctly handled. Go ahead and commit, push and tag this experiment

```
git ac -m "Size 36 convNet"
dvc push
git tag -a "36-convnet" -m "Size 36 convNet"
git push --follow-tags
```
Of course, all of these commands could be combined in a simple alias, but I use them here as a reminder of what is happening. If you would like a fun shortcut, add the following to your .git_config in the HOME directory. 

```
[alias]
        dv = "!f() { \
                   git add -A && git commit -m $1 && dvc push && git tag -a $2 -m $1 && git push --follow-tags; \
                   }; f" 
```
and run with `git dv "Commit_message" "tag-name"`.


* **git commit, dvc push**
* **Run the model with `dvc repro`**

Do you remember what the baseline model's accuracy was? Luckily, we can quickly check. 

```
dvc metrics show -T
```

If you're anything like me, you'll see that the model did a little better. It may have even been faster.

Let's make more changes to the `Net()` model in `convNet.py`, for example by adding many more fully connected weights. Try running repro evaluate. We should get a message saying that nothing has changed, so there's no need to run the evaluation. That's because now we're only changing our model file, which is not under DVC control or part of the pipeline. We need to update the pipeline to reflect our refactored structure. `dvc add` the models/ directory, and re-run the `dvc run` commands above, with `-d src/models/` included as a flag. We will see the new model's performance - it's even better for me. Commit, push and tag it. Compare metrics, and I have

```
working tree:                                                           
        acc.metric:
                6.829999999999999716e+01
                5.750573873519897461e+00
36-convnet:
        acc.metric:
                5.759000000000000341e+01
                5.656264781951904297e+00
baseline:
        acc.metric:
                5.602000000000000313e+01
                7.337175607681274414e+00
huge-convnet:
        acc.metric:
                6.829999999999999716e+01
                5.750573873519897461e+00
```

My very wide, 2-layer ConvNet is giving 68% accuracy, in 5.7 seconds. Go ahead and change some more of the hyperparameters, running `dvc repro evaluate.dvc` and commiting and tagging each experiment. 

---
**Some hard-won lessons on git vs. dvc control**

In general, feel free to put everything under git control, and then subsequently add it as data with `dvc add`. However, be aware of some catches:
* DVC will complain if it's already under the control of git. Simply remove it first from git control with `git rm (-r) --cached <file (or folder) name>`
* Any dependencies of `dvc run` should be under dvc control (i.e. `dvc add`ed), or the dependency will not be tracked, and it is superfluous to include it in `dvc run`. Only `-o` flagged files/folders will be automatically tracked, not `-d` flagged files/folders
* You may accidentally `git commit` some large data. Every subsequent push to GitHub will throw a complaint about the size of the commit. Breathe, relax, and run `git filter-branch --tree-filter 'rm -rf path/to/your/file-or-folder' HEAD` on the offending data folder or file. See [this SO answer](https://stackoverflow.com/questions/10622179/how-to-find-identify-large-commits-in-git-history) for further help on this mistake.
---

I found a good improvement from increasing the number of training epochs to > 5. DVC is very good at making these steps easy to run, and ensuring that no data is lost along the way. But its metric comparison leaves a lot to be desired. We can insert some easy functions provided by Weights & Biases to visualise the training over epochs, and between models.



## Visualising Performance 

To begin, ensure that Weights & Biases is installed with `pip install --user wandb`. You will also need run wandb login from the command line, ensuring you have a W&B account (by signing up at www.wandb.ai).

Now we need to make the following additions to our `train.py` script:
1. `import wandb` in the prelogue
2. `wandb.init(project="convnet-toy")` before the training loop
3. `wandb.watch(model, log="all")` in the train() function
4. `wandb.log({"train loss": train_loss, "val loss": val_loss, "val accuracy": val_acc})` at the end of each training epoch

Any results from changes we make to the architecture or hyperparameters will be tracked in W&B. Let us call this "manual tuning". It doesn't require any more code updates than these four steps. Go ahead and run `dvc repro train.dvc` to see the results be sent to W&B. You can see my output with https://app.wandb.ai/murnanedaniel/convnet-toy

Now that we have a more sophisticated way to visualise and identify better-performing models, there is also a more sophisticated way to reproduce them. I suggest two convenient habits:
1. During training, save the checkpoint of each epoch in a `/checkpoints` folder. At the end of training, save the best-performing model checkpoint to W&B with `wandb.save('/checkpoints/<best-model-epoch>')`. Then, given a run ID, the best model and its HP configurations are quickly available for inference with `best_model = wandb.restore('<best-model-epoch>', run_path="murnanedaniel/convnet-toy/<run ID>"`.
2. For closer analysis, and to fully reproduce the run, ensure that the `/checkpoints` folder is tagged with `-o` in `dvc run` and git committed. Then, run `wandb restore <run ID>` from the command line. This checks out the git commit associated with a particular run ID, then we can run `dvc checkout` to restore all the input data and model and checkpoint files associated with the run. 
**CHECK THAT THIS WORKS! THERE MIGHT BE AN ISSUE WITH THE COMMIT BEING ONE STEP BEHIND...**

## Hyperparameter Optimisation

Broadly, our model can be described as Convolution 1 with kernel size K<sub>1</sub>, hidden layer size H<sub>1</sub>, Convolution 2 with kernel size K<sub>2</sub>, hidden layer size H<sub>2</sub>, two fully connected layers with H<sub>3</sub> and H<sub>4</sub> hidden features, and an output layer. Thus we have 6 interesting hyperparameters (HPs) within the model, not to mention several training HPs, including the learning rate, momentum, and number of epochs. Starting from this set, we have a 9-dimensional space in which to find the best model. Rather than searching this manually, we can ask W&B to select combinations of these 9 HPs and pass them to the model class and training method. 

We add the HPs to the class construction of Net(), as 

```
class Net(nn.Module):
    def __init__(self, kern_1 = 5, hidden_dim_1 = 6, kern_2 = 5, hidden_dim_2 = 16, hidden_dim_3 = 120, hidden_dim_4 = 84, output_dim = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim_1, kern_1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kern_2)
        self.flat_num = int(np.ceil( (np.ceil( (32 - kern_1) / 2) - kern_2 ) /2 ))
#         print(self.flat_num)
        self.fc1 = nn.Linear(hidden_dim_2 * self.flat_num * self.flat_num, hidden_dim_3)
        self.fc2 = nn.Linear(hidden_dim_3, hidden_dim_4)
        self.fc3 = nn.Linear(hidden_dim_4, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

where the `num_flat_features` method is borrowed from the PyTorch NN tutorial as a handy automation of handling dimension. 


### Option 1: Call the Sweep from within a Python function, sweep.py

To have W&B select the HPs, I have found it convenient to handle them as dictionaries of configurations. For example, we have a model dictionary that is defined as

```
m_dic = ["kern_1", "hidden_dim_1", "kern_2", "hidden_dim_2", "hidden_dim_3", "hidden_dim_4"] # Define the HPs given by W&B
m_configs = {k:wandb.config.get(k,0) for k in m_dic} # Retrieve the HPs from W&B
m_configs = {**m_configs, 'output_dim': 10} # Manually specify any HPs not part of the sweep
model = Net(**m_configs).to(device) # Initialise the model, and send it to the GPU
```

We can do the same for the optimisation HPs

```
o_dic = ["lr", "momentum"]
o_configs = {k:wandb.config.get(k,0) for k in o_dic} 
optimizer = optim.SGD(model.parameters(), **o_configs)
```
and finally get the number of epochs with 
``` 
for epoch in range(wandb.config.get("n_epochs", 0)): ...
```

* Incorporate sweep agent in training.py

The main shift is that we now need `main()` to be in charge of running the sweep agent, which calls `train()`. Our `main()` should now consist of 

```
# Parse the command line
args = parse_args()
        
# Load config YAML
with open(args.config) as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        
# Instantiate WandB sweep ID
sweep_id = wandb.sweep(sweep_config, entity= "murnanedaniel", project= "edge_classification_sweep")
    
# Run WandB weep agent
wandb.agent(sweep_id, function=train)
```

### Option 2: Run the Sweep from the command line

**This is strongly suggested** if you intend to run the sweeps in a distributed way (multi-GPU or multi-node).

In this scenario, we hand the HPs to the file by assuming they will be passed at the command line as flags. Thus, we need to include `argparse` flags, such as

```
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('sweep_cli.py')
    add_arg = parser.add_argument
    add_arg('--hidden_dim_1', type=int)
    add_arg('--hidden_dim_2', type=int)
    add_arg('--hidden_dim_3', type=int)
    ...
```

and then build the model/optimiser config dictionaries from these. This is because, to run the sweep, we run

```
wandb sweep /src/config/sweep_config.yaml
wandb agent <sweep_ID>
```

which will call `python src/sweep_cli.py --hidden_dim_1=420 --hidden_dim_2=... etc.`

What is this sweep_config concept? It stores all of the information required for W&B to run the sweep to our specifications. For example, I'm using the following sweep configuration, in a sweep_config.yaml file:

```
(program: sweep_cli.py )
method: random
name: ConvNet Sweep
parameters:
    hidden_dim_1: 
        min: 4 
        max: 1000
    kern_1:
        min: 2
        max: 8
    hidden_dim_2: 
        min: 4 
        max: 1000
    kern_2:
        min: 2
        max: 8
    hidden_dim_3: 
        min: 20 
        max: 4000
    hidden_dim_4: 
        min: 20 
        max: 1000
    lr:
        distribution: log_normal
        max: -2.3
        min: -11.5
        mu: -6.9
        sigma: 1.5
    momentum:
        min: 0.1
        max: 0.9
    n_epochs:
        min: 2
        max: 10
```

Note that this will be a "dumb" sweep - it is randomly sampling the HP space. We will make it "smart" later.

## An Opinionated Way to Handle DVC with W&B

![](https://imgs.xkcd.com/comics/standards.png)

* Run the sweep on interactive node
Now, we add a `dvc run` for the sweep. Note that we can't have the same output file as the regular train script, so let's change the model output to `cifar_net_sweep.pth`.

* Submit it as batch job... debug this

### Update the Dataset

* Add some transformations / augmentations to prepare.py
* Run this into the pipeline
* Show when this runs, and when it doesn't need to


## Important Points
