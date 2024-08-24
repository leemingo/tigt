Our code is developed for the paper titled "TOPOLOGY-INFORMED GRAPH TRANSFORMER", submitted to ICLR 2024.

Our implementation has been made compatible in a Python environment similar to GraphGPS.
(https://github.com/rampasek/GraphGPS)

There is example to run the code : 
python main.py --cfg configs/TIGT/peptides-func-TIGT.yaml --repeat 1 seed 10 wandb.use False

The config files for the GPS and GRIT models work within their respective environments as uploaded on GitHub. (https://github.com/rampasek/GraphGPS , https://github.com/liamma/grit )

In the case of the CSL data, it is assumed that the values for nodes and edges are all uniformly set to 1.