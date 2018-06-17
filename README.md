# Python module for adaptive umbrella sampling

This module can be used to perform adaptive umbrella sampling of a multi-dimensional potential of mean force. The
algorithm involves::

1) calculate the free energy landscape
2) Among existing windows, select windows with E < E_max
3) For each selected window, generate 3^N-1 neighbor windows
4) Sample new windows, then go to 1) or stop if no new windows can be found

For more details about the algorithm: See

Self-Learning Adaptive Umbrella Sampling Method for the Determination of Free Energy Landscapes in Multiple Dimensions (Wojtas-Niziurski†, Meng, Roux, Bernèche, 2013)
[(https://doi.org/10.1021/ct300978b)](https://doi.org/10.1021/ct300978b)

