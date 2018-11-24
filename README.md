![Lightpost](https://github.com/dlsucomet/Lightpost/blob/master/misc/logo2.png)

## 
Lightpost is an automated neural networks training interface written in Python and runs on top of the [PyTorch](https://github.com/pytorch/pytorch) framework. It was developed with fast and seamless prototyping in mind. It is made up of four core modules:

* ```lightpost.engine``` - An automated neural networks training engine that enables quick training pipelines.
* ```lightpost.datapipe``` - An automated data pipeline that seamlessly converts data into trainable batches using preprocessing pipelines.
* ```lightpost.estimators``` - Provides ready-built models adaptable to different tasks, integrateable and extensible using PyTorch.
* ```lightpost.utils``` - Provides various utility functions for general tasks and specialized (NLP, Vision) tasks.

The Lightpost Project is built with inter-operability at its core. It is friendly and plays well with your existing frameworks and modules.

Lightpost is currently in development. Be sure to check the repo freely for updates!

## Prerequisites and Installation
Lightpost depends on the following packages:
* PyTorch 0.4.x
* TorchText 0.4.x
* TorchVision 0.4.x

Clone the repository to your machine in the directory of your projects.
```
git clone https://github.com/dlsucomet/Lightpost.git
```

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
engine = Engine(pipeline=pipe, model=model, criterion='cross_entropy', optimizer='adam')

# Train the model for 1000 epochs, printing the losses and accuracies every 100 epochs
engine.fit(epochs=1000, print_every=100, disable_tqdm=True)

# Save the model's weights for use later on
engine.save_weights('model.pt')

```

Lightpost's specialized data pipelines can be used for more special cases. Here is an example workflow for an NLP task, sentiment classification:

```python
from lightpost.datapipe import Textpipe
from lightpost.estimators import LSTMTextClassifier
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

## Changelogs
**Version 0.0.3**
* Added ```lightpost.estimators``` API with two models.
* Streamlined the ```lightpost.engine``` and ```lightpost.datapipe``` APIs.
* Added NLP support in ```lightpost.utils.text``` for standard NLP preprocessing functions.
* Fixed optimization bugs when training using the engine.