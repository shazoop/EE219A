Final Project Files Overview:

Goal: simulate a two-layer model of simple/complex cells of the primary visual cortex - V1 - and
try to reproduce response properties of V1 cells.

Two models were simulated:
1) Model 1: Simple neurons/FE: artificial neurons with sigmoid activation. neurons as well as synaptic weights simulated in
tandem. FE used to solve.
1) Model 2: Spiking neurons/Implicit Methods: used Fitzhugh-Nagumo model to simulate spiking of neurons. synaptic weight
now update in discrete steps

Included are: 

-Report of the final project
-Files:
  -Data:
	-data_generator.py: module for generating movie sequences of MNIST digits, the input data
  -FE:
	-FE_DAE.py: DAE of Model 1 - simple neurons
	-FE_DAEsolver.py: solver for Model 1 using Foward Euler
	-EE219A_run-Clean.ipynb: Jupyter notebook of code used to simulate Model 1
  -V1full:
 	-V1_DAE-Clean.py: DAE of Model 2 - spiking neurons using Fitzhugh-Nagumo
	-NR.py: implementation Newton-Raphson. no limiting. initiallize automatically to previous unknown.
	-DAE_solver.py: implements FE/BE/Trap. depends on NR.py
	-Fitz_Nag.py: DAE of Fitzhugh-Nagumo model. was used for intial testing/tuning of FN parameters.
	-EE219A_FullDAE-Clean.ipynb: Jupyter notebook of code used to simulate Model 2
 