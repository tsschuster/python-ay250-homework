Assignment 4 for AY250 - Paralellism

=======================

Summary of results:

As expected, the execution time for serial Monte Carlo increased linearly with the number of darts thrown for all times tested.

The execution time for multiprocessing Monte Carlo was constant around ~50ms for low numbers of darts thrown (<~ 10^5-6), indicating ~50ms of "overhead" for using the pool program. For higher numbers of darts it approached linear behavior, with total time ~1/4 that of serial Monte Carlo - the expected speed up for my machine with 4 cores.

The execution time for dask Monte Carlo displayed the same qualitative trends as multiprocessing Monte Carlo, although was much (~10x) faster in the linear regime. 

This ~10x speedup could also be had for the serial & multiprocessing functions if we used numpy to generate an array of random x & y values, instead of a for loop. (This method is commented out in my code for perform_MC()).

All simulations were run on a MacBook Pro with 2.8 GHz Intel Core i7, 4 “cores”.