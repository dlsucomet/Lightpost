![Lightpost](https://github.com/dlsucomet/Lightpost/blob/master/misc/logo2.png)

## 
Lightpost is an automated neural networks training system and pipeline that aims to make research and model building as seamless and headache-free as possible. It is made up of two main components:

* ```lightpost.engine``` - An automated training system. Build and register a model, then prepare a training configuration. The system will load and preprocess your data, configure the pipeline and loggers, and train the model for you. Reduces your time wrestling with code so you can focus on reading more papers.
* ```lightpost.layers``` - A high-level neural networks framework inspired by [Keras](https://github.com/keras-team/keras). Defines stackable layers and modules that reproduce models from state-of-the-art deep learning papers. Train Generative Adversarial Networks and ULMFiT models with ease.

The Lightpost Project is built on top of the [PyTorch](https://github.com/pytorch/pytorch) Framework, and ensures inter-operability and extensibility with your existing Python packages.

Lightpost is currently in development. Be sure to check the repo freely for updates!

## Prerequisites and Installation
Lightpost depends on the following packages:
* PyTorch 0.4.x
* TorchText 0.4.x
* TorchVision 0.4.x

Clone the repository to your machine
```
git clone https://github.com/dlsucomet/Lightpost
```

Configure your settings and run ```main.py```
```
$ python3 main.py
```

## Usage
**Using the Engine.** After creating a model in Lightpost and registering it in ```lightpost.models``` you can now configure ```main.py``` with your own settings, for example:

```
from lightpost.engine import Engine

config = {
  'dataset' : 'datasets/imdb.csv'
  'train_split' : 0.8
  'target_index' : 0
  'model' : 'GRUTextClassifier',
  'model_params' : { 'hidden_dim' : 128, 'bidirectional' : true },
  'lr' : 1e-4,
  'epochs' : 100,
  'use_cuda' : true,
  'use_logging' : true,
  'save_weights_to' : 'weights/gruweights.h5'
}

engine = Engine(config=config)
engine.train()

```

Then run the script in your terminal.

```
$ python3 main.py
```

**Using the Layers Framework.** The ```lightpost.layers``` framework is used just like any other deep learning framework. You can use it to build models that are trainable with the ```lightpost.engine``` interface.

```
from lightpost.layers import Attention, LSTM, Linear
from lightpost.layers.modules import Sequential

model = Sequential(name='AttentionLSTM')
model.stack(LSTM(cells=128, bidirectional=True, input_shape=X_train.shape))
model.stack(Attention(step_dim=100))
model.stack(Linear(64, activation='relu'))
model.stack(Linear(3, activation='sigmoid'))
model.register()
```

You can now use ```AttentionLSTM``` in a training configuration. You can also use the framework as an extension to PyTorch.

```
from lightpost.layers.modules import Transformer

model = Transformer(vocab_size=vocab_size, max_token_len=max_toke_len, ...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

...
```

## Changelogs
*Coming Soon*