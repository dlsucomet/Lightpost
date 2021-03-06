![Lightpost](https://github.com/dlsucomet/Lightpost/blob/master/misc/logo2.png)

## 
Lightpost is an automated neural networks training interface written in Python and runs on top of the [PyTorch](https://github.com/pytorch/pytorch) framework. It was developed with fast and seamless prototyping in mind. It is made up of four core modules:

* ```lightpost.engine``` - An automated neural network training engine.
* ```lightpost.datapipe``` - An automatic data preprocessing pipeline. Seamlessly converts data into trainable batches.
* ```lightpost.estimators``` - Provides prewritten models for different tasks, integrateable and extensible using PyTorch.
* ```lightpost.utils``` - Provides various utility functions for general and specialized (NLP, Vision, etc) tasks.

The Lightpost Project is built with inter-operability at its core. It is friendly and plays well with your existing frameworks and modules.

Lightpost is currently in development. Be sure to check the repo freely for updates!

## Prerequisites and Installation
Lightpost runs on Python 3.6 and depends on the following packages:
* PyTorch 0.4.x
* TorchText 0.4.x
* TorchVision 0.4.x
* Tensorflow 1.12 (for Tensorboard Support)
* TensorboardX 1.4

Clone the repository to your machine in the directory of your projects.
```bash
git clone https://github.com/dlsucomet/Lightpost.git
```

Then run the setup script.
```bash
python3 setup.py install
```

This should install the ```lightpost``` package, which is trackable by pip. To uninstall, run ```pip3 uninstall lightpost``` just like any other pip-based package.

## Usage
Here are a few examples on Lightpost-powered workflows. For a detailed demo, check out our [demo notebook](https://github.com/dlsucomet/Lightpost/blob/master/Lightpost%20Test.ipynb)!

This is an example workflow that uses the ```lightpost.engine```, ```lightpost.estimators```, and ```lightpost.datapipe``` interfaces for simple classification. 

```python
from lightpost.datapipe import Datapipe
from lightpost.estimators import MLPClassifier
from lightpost.engine import Engine

from sklearn.datasets import load_iris
d = load_iris() # We'll use the iris dataset

# Construct a pipeline from the iris dataset then build the engine
pipe = Datapipe(d.data, d.target)
model = MLPClassifier(input_dim=4, hidden_dim=128, output_dim=3, num_layers=3)
engine = Engine(pipeline=pipe, model=model, criterion='cross_entropy', optimizer='adam', use_tensorboard=True)

# Train the model for 1000 epochs, printing the losses and accuracies every 100 epochs
engine.fit(epochs=1000, print_every=100, disable_tqdm=True)

# Save the model's weights for use later on
engine.save_weights('model.pt')

```

The ```use_tensorboard``` option allows you to use Tensorboard to log the engine's training statistics. Using ```use_tensorboard=True``` will create a directory named ```runs``` in the directory you are working in. To run Tensorboard, run ```tensorboard --logdir runs``` in a terminal.

Lightpost's specialized data pipelines can be used for more special cases. Here is an example workflow for an NLP task, sentiment classification:

```python
from lightpost.datapipe import Textpipe
from lightpost.estimators import LSTMClassifier
from lightpost.engine import Engine

# Automatically preprocesses the text dataset
pipe = Textpipe(path='data/train.csv', text='comments', target='sentiment', maxlen=50)

# Pretrained word embeddings can also be used in the pipeline
pipe = Textpipe(path='data/train.csv', text='comments', target='sentiment', maxlen=50, 
                pretrained_embeddings=True, embed_path='vectors/wiki.en.vec', embed_dim=300)

# Create the LSTM Text Model with pretrained embeddings automatically loaded using a datapipe
model = LSTMClassifier(pretrained=pipe.embedding, embedding_dim=pipe.embed_dim, 
                       hidden_dim=256, output_dim=2, bidirectional=True, recur_layers=2, 
                       recur_dropout=0.2, dropout=0.5)

# Construct the engine with a learning rate decay scheduler
engine = Engine(pipeline=pipe, model=model, criterion='cross_entropy', optimizer='adam', scheduler='plateau')
engine.fit(epochs=100, print_every=10, disable_tqdm=False)

engine.save_weights('model.pt')

```

Lightpost, being built on top of PyTorch, accepts custom models. Here is an example workflow:

```python
import torch
import torch.nn as nn

from lightpost.datapipe import Datapipe
from lightpost.estimators import MLPClassifier
from lightpost.engine import Engine

from sklearn.datasets import load_iris

# Define your model
class MLPClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MLPClassifier, self).__init__()

    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, output_dim)

  def forward(self, X):
    out = torch.relu(self.fc1(X))
    out = torch.relu(self.fc2(out))
    out = torch.sigmoid(self.output(self.fc3(out)))
    return out

# Load the iris dataset
d = load_iris()

# Construct a pipeline from the iris dataset then build the engine
pipe = Datapipe(d.data, d.target, batch_size=32)
model = MLPClassifier(input_dim=4, hidden_dim=128, output_dim=3)
engine = Engine(pipeline=pipe, model=model, criterion='cross_entropy', optimizer='adam')

# Train the model for 1000 epochs, printing the losses and accuracies every 100 epochs
engine.fit(epochs=1000, print_every=100, disable_tqdm=True)

# Save the model's weights for use later on
engine.save_weights('model.pt')

```

## Features We're Working On
Before we increment the version counter, we'll make sure that some important features are included. You might see these features in alpha stage in nightly builds, which might break your setups, so be careful!

**For Version 0.1 Release**
* CUDA support with automated mixed precision (FP16) training to double/maximize GPU memory
* Computer Vision support in ```lightpost.datapipe```, called ```Imagepipe```
* Computer Vision utility functions under ```lightpost.utils.vision```
* Support for one-shot/few-shot training pipelines *(currently in alpha stage)*

## Releases and Contribution
**Release Cycle.** Lightpost is under a non-regular release cycle. It's currently in Alpha state, where bugs are expected when you try to forcefully do things Lightpost isn't supposed to.

If you encounter bugs, please report them in our [GitHub Issues](https://github.com/dlsucomet/Lightpost/issues) tracker.

**Feature Requests.** For feature requests, please drop by to our [GitHub Issues](https://github.com/dlsucomet/Lightpost/issues) tracker and we'll discuss with you there.

## Acknowledgements
The Lightpost Project was born out of a need for seamless experimentation on multiple models. In research, we often test models multiple times with different hyperparameters and settings. Jupyter notebooks would rarely cut the job once we're working on so many things at once. A dynamic training framework that can be written in the form of small scripts was needed.

Special thanks to Daniel Stanley Tan, whose seamless GAN training scripts provided the initial inspiration for Lightpost, as well as to Briane Paul Samson for the constant support during development. Lightpost, while mainly maintained by a group of researchers, owes itself to helpful contributions from the community in various forms. Acknowledgements are also on the way for the researchers of the DLSU Center for Complexity and Emerging Technologies, its lab head Jordan Aiko Deja, and its faithful leadership team.

Project Lightpost is developed, maintained, and managed by the DLSU Machine Learning Group. 