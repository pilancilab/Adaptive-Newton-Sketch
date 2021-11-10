# Readme

This implements the code for __Adaptive Newton Sketch:  Linear-time Optimization with QuadraticConvergence and Effective Hessian Dimensionality__.



This is partially based on the implementation of Newton Sketch in [this github repo](https://github.com/huisaddison/newton-sketch).



# Datasets

All dataset files can be downloaded from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). To generate kernel matrices, run the following lines:

```
python dataset-a8a-kernel.py
python dataset-phishing-kernel.py
python dataset-w7a-kernel.py
```



# Reproduce results

For GradientDescent and NAG, run Newton-Sketch-ada first and run

```
python gen_loss_ref.py
```

Then, place the *.p files into the corresponding folder of datasets.



To reproduce the results in the paper, run the following lines.

- rcv1

```
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --m 100 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 2 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --m 100 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 1 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --m 800 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --m 800 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --mu 1e-3 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --mu 1e-3 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name rcv1 --n 1e4 --d 47236 --mu 1e-3 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```

- mnist

```
python3 main.py --data_name MNIST --n 3e4 --d 780 --m 100 --mu 1e-1 --optim Newton-Sketch-ada --shuffle --lbdtol 0.5 --lbdpow 2 --lbdtol2 6 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name MNIST --n 3e4 --d 780 --m 100 --mu 1e-1 --optim Newton-Sketch-ada --shuffle --lbdtol 0.5 --lbdpow 2 --lbdtol2 6 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name MNIST --n 3e4 --d 780 --m 800 --mu 1e-1 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name MNIST --n 3e4 --d 780 --m 1600 --mu 1e-1 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name MNIST --n 3e4 --d 780 --mu 1e-1 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name MNIST --n 3e4 --d 780 --mu 1e-1 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name MNIST --n 3e4 --d 780 --mu 1e-1 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```

- gisette

```
python3 main.py --data_name gisette --n 3e3 --d 5e3 --m 10 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 2 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name gisette --n 3e3 --d 5e3 --m 10 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 2 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name gisette --n 3e3 --d 5e3 --m 400 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name gisette --n 3e3 --d 5e3 --m 400 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name gisette --n 3e3 --d 5e3 --mu 1e-3 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name gisette --n 3e3 --d 5e3 --mu 1e-3 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name gisette --n 3e3 --d 5e3 --mu 1e-3 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```

- realsim

```
python3 main.py --data_name realsim --n 5e4 --d 20958 --m 100 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 2 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name realsim --n 5e4 --d 20958 --m 100 --mu 1e-3 --optim Newton-Sketch-ada --shuffle --lbdtol 2 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name realsim --n 5e4 --d 20958 --m 800 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name realsim --n 5e4 --d 20958 --m 3200 --mu 1e-3 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name realsim --n 5e4 --d 20958 --mu 1e-3 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name realsim --n 5e4 --d 20958 --mu 1e-3 --optim GradientDescent --shuffle --max_iter 8000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name realsim --n 5e4 --d 20958 --mu 1e-3 --optim NAG --shuffle --max_iter 8000 --data_dir PATH_TO_DATASETS --use_line_search
```

- epsilon

```
python3 main.py --data_name epsilon --n 5e4 --d 2000 --m 100 --mu 1e-1 --optim Newton-Sketch-ada --shuffle --lbdtol 1 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name epsilon --n 5e4 --d 2000 --m 100 --mu 1e-1 --optim Newton-Sketch-ada --shuffle --lbdtol 1 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name epsilon --n 5e4 --d 2000 --m 800 --mu 1e-1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name epsilon --n 5e4 --d 2000 --m 3200 --mu 1e-1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name epsilon --n 5e4 --d 2000 --mu 1e-1 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name epsilon --n 5e4 --d 2000 --mu 1e-1 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name epsilon --n 5e4 --d 2000 --mu 1e-1 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```

- a8a-kernel

```
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --m 10 --mu 1e1 --optim Newton-Sketch-ada --shuffle --lbdtol .5 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --m 10 --mu 1e1 --optim Newton-Sketch-ada --shuffle --lbdtol .5 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --m 100 --mu 1e1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --m 800 --mu 1e1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --mu 1e1 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --mu 1e1 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name a8a-kernel --n 1e4 --d 1e4 --mu 1e1 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```

- w7a-kernel

```
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --m 10 --mu 1e1 --optim Newton-Sketch-ada --shuffle --lbdtol .5 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --m 100 --mu 1e1 --optim Newton-Sketch-ada --shuffle --lbdtol .5 --lbdpow 1 --max_iter 200 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --m 100 --mu 1e1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type SJLT
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --m 800 --mu 1e1 --optim Newton-Sketch --shuffle --max_iter 800 --data_dir PATH_TO_DATASETS --sketch_type RRS
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --mu 1e1 --optim Newton --shuffle --max_iter 200 --data_dir PATH_TO_DATASETS
```

```
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --mu 1e1 --optim GradientDescent --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
python3 main.py --data_name w7a-kernel --n 12000 --d 12000 --mu 1e1 --optim NAG --shuffle --max_iter 3000 --data_dir PATH_TO_DATASETS --use_line_search
```



# Plot figures

To plot figures, run `python plot_figure_DATASET.py`. This will generate figures that are based on the program outputs.