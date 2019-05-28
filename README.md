
<img src="https://raw.githubusercontent.com/MalikLe/flussPred/tunning/logo_smart.png" title="Fluss" alt="Fluss">

# Fluss

Fluss smart is a python application for predicting flooding scenarios all over the world.

## Installation

### Clone

Clone this repo to your local machine using https://github.com/fhugard/flussPred.git

### Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies.

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import math
import datetime
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator
```

## Usage
Set the parameters in main.py
```python
lstm(s_window=90, l_fwd=30, n_units=16, n_epochs=1, batch_size=32)
```
Then compile

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
