[![Build Status](https://travis-ci.org/danijoo/adaptiveumbrella.svg?branch=master)](https://travis-ci.org/danijoo/adaptiveumbrella)

# Python module for adaptive umbrella sampling

This module can be used to perform adaptive umbrella sampling of a multi-dimensional potential of mean force. The
algorithm involves::

1) calculate the free energy landscape
2) Among existing windows, select windows with E < E_max
3) For each selected window, generate 3^N-1 neighbor windows
4) Sample new windows, then go to 1) or stop if no new windows can be found

For more details about the algorithm, see

Self-Learning Adaptive Umbrella Sampling Method for the Determination of Free Energy Landscapes in Multiple Dimensions (Wojtas-Niziurski, Meng, Roux, Bernèche, 2013)
[(https://doi.org/10.1021/ct300978b)](https://doi.org/10.1021/ct300978b)


![Example](https://raw.githubusercontent.com/danijoo/adaptiveumbrella/master/examples/example.gif)

## Installation

Adaptiveumbrella supports python version 3.4+ and can be installed via pip or manually:

```bash
pip install --user git+https://github.com/danijoo/adaptiveumbrella.git
```

Alternative:
```bash
git clone https://github.com/danijoo/adaptiveumbrella.git
cd adaptiveumbrella
python setup.py install --user
```


## Usage

Implement the UmbrellaRunner class according to your needs: 

```python
from adaptiveumbrella.runner import UmbrellaRunner

class MyUmbrellaRunner(UmbrellaRunner):
  pass
```

within the class we need to define two methods. First we have to define how the simulation windows should be sampled:

```python
  def simulate_frames(self, lambdas, frames):
    """ Run simulations for all passed lambda steps. `lambdas` is a dictionary where each key
    is a tuple of coordinates in the phase space and each value are the lambda values of the root
    from which this frame should be created. `frames` is an identical dict, but with indeces of the pmf
    numpy array defining the phase space """
    pass
```

then, we have to implement a method that updates the pmf:
 
```python
  def calculate_new_pmf(self):
  """ This is called after `simulate_frames` and should calculate the new PMF. return value must be a numpy
  array of similar dimensions then the lambda states."""
  pass
```

Finally, we can instantiate the class, pass the configuration variables and start the simulations:

```python
runner = MyUmbrellaRunner()

# 2 dimensional phase space ranging from -3 to 3 in both dimensions
# lambda spacing is 0.2 in x and y
runner.cvs = np.array([
    (-3, 3, 0.2),
    (-3, 3, 0.2),
])
# initial lambda coordinates
runner.cvs_init = (1.4, -1.4)

# initial energy for finding new frames
runner.E_min = 10

# max. energy before sampling stops
runner.E_max = 100

# energy change between steps
runner.E_incr = 10

# max. number of iterations before stopping
runner.max_iterations = 100

# let's go
runner.run()
```

