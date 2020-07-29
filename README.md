# [Incorporating physical constraints in a (deep) probabilistic machine learning framework for coarse-graining dynamical systems](https://www.sciencedirect.com/science/article/pii/S0021999120304472)

## Highlights:
- A generative model for the automated discovery of CG dynamics.
- The target density is augmented by virtual observables which reflect physical constraints.
- The incorporation of physical constraints leads to a reduction of the training data.
- A probabilistic formulation that is capable of quantifying predictive uncertainty.
- Full reconstruction of futures of the entire FG state vector as well as any FG observable.

## Content:
The folder Particle contains some of the training data, code and results for the Advection-Diffusion and Burgers' example from the paper. 
The folder Pendulum contains some of the training data, code and results for the pendulum example from the paper.

## Dependencies
For the particle example:
- Tensorflow 1.13.1
- mpi4py

For the pendulum example:
- Tensorflow 2.0

## Particle example
The proposed framework is applied to a system of moving particle. We are able to extract meaningful CG dynamics as well as to do interpolative and extrapolative predictions.

![overview](https://raw.githubusercontent.com/SebastianKaltenbach/PhysicalConstraints_ProbabilisticCG/master/Example_Burgers.pdf)

To visualize the results, please use the file Prediction.ipynb

To start the training process, please run mpi_train.py using mpirun


## Pendulum example
The proposed framework is applied to a series of images of a nonlinear pendulum. We are able to extract the two-dimensional dynamics as well as a coarse-to-fine mapping. 

![overview](https://raw.githubusercontent.com/SebastianKaltenbach/PhysicalConstraints_ProbabilisticCG/master/pendulum_animated.gif)

To visualize the results, please use the file Prediction.ipynb

To start the training process, please run training.py 


## Citation
If this code is relevant for your research, please consider citing:
```
@article{kaltenbach2020incorporating,
  title={Incorporating physical constraints in a deep probabilistic machine learning framework for coarse-graining dynamical systems},
  author={Kaltenbach, Sebastian and Koutsourelakis, Phaedon-Stelios},
  journal={Journal of Computational Physics},
  year={2020},
  publisher={Elsevier},
  doi = "https://doi.org/10.1016/j.jcp.2020.109673",
  url = "https://www.sciencedirect.com/science/article/pii/S0021999120304472"
}
```
