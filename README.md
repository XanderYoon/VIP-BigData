# VIP-Big Data in Quantum Mechanics (Fall 2023)
Documentation of my experience with the Medford Research Group (Big Data in Quantum Mechanics). 

# 1.6

## 1.6.1, PLT

1 import numpy as np
2 import pylab as plt 
3 import os
4 from sklearn.metrics import mean_absolute_error
5 
6 S = [2, 5, 10, 16, 24, 27, 35, 50]
7 data = list(transformer(z) for z in S)
8 
9 # How do I make it faster so I don't have multiple for loops
10 xList = [x[0] for x in data]
11 yList = [y[1] for y in data]
12 
13 plt.xlabel('Guess')
14 plt.ylabel('Iterations')
15 plt.plot(xList, yList, '-o', color='black')
16 
17 plt.savefig("Collatz_exercise.png", dpi=300)
18 
19 with open('Collatz.out', 'w') as file:
20     for guess, iterations in data:
21         file.write(f'Initial Guess: {guess}, Iterations: {iterations}\n')

## 1.6.2, String

1 def print_even(string_in):
2     if (len(string_in) == 0):
3         raise Exception("Empty String")
4     
5     evenChar = ""
6     for x in range(1, len(string_in), 2):
7         evenChar += string_in[x] + " "
8     print(evenChar)
9     
10 '''
11 def print_even(string_in):
12     if (len(string_in) == 0):
13         raise Exception("Empty String")
14     evenChar = string_in[1::2]
15     print(evenChar)
16 '''
17 
18 print_even("Python")
19 # print_even("")
20 
21 S = ["Gojackets", "Call me Ishmael", "ILoveChBE"]
22 for x in S:
23     print_even(x)

## 1.6.3, Numbers

1 def get_tax(income):
2     finalTax = 0
3     if (income > 10000):
4         finalTax = (income - 10000) * 0.1
5     if (income > 25000):
6         finalTax += (income - 25000) * 0.13
7     return finalTax
8 
9 get_tax(58500)

## 1.6.4, Reverse

1 def print_reverse(number):
2     number_reverse = str(number)[::-1]
3     print(number_reverse)
4 
5 print_reverse(6572)
6 
7 S = [1, 15, 658, 2940, 44112]
8 for number in S:
9     print_reverse(number)

## 1.6.5, MAE

1 x = [-7,1,5,2,9,-2,0,1]
2 y = [-6,4,4.5,2,11,-2.1,1,3]
3 
4 mean_absolute_error(x, y)

# 2.5

## 2.5.1, CaLTA

1 atom = read('Data/ACAJIZ.cif')
2 atom.get_global_number_of_atoms()
3 cif.get_atomic_numbers()
4 print(cif.symbols.species())
5 print(cif.get_volume())
6 
7 from ase.calculators.emt import EMT
8 from ase.units import Bohr, Hartree, mol, kcal, kJ, eV
9 from ase.optimize import BFGSLineSearch
10 
11 print(f"Energy before relaxation: ", atom.get_total_energy())
12 
13 atom.set_calculator(EMT())
14 optimizer = BFGSLineSearch(atom, trajectory='opt.traj', logfile='opt.log')
15 optimizer.run(fmax=0.3)
16 
17 print(f"Energy after relaxation: ", atom.get_total_energy())

# 3.3

## 3.3.1, Test.py

- ssh ayoon37@login-ice.pace.gatech.edu → Type in password
- cd scratch
- mkdir Training
- nano test.py → print('Hello World') → ctrl+x, y, ¡enter¿

## 3.3.2, Bash Script

1 #!/bin/bash
2 # SBATCH --job-name=test_job
3 # SBATCH --output=test.out
4 # SBATCH --error=test.err
5 # SBATCH --nodes=1
6 # SBATCH --ntasks=1
7 # SBATCH --cpus-per-task=1
8 # SBATCH --time=5:00
9 cd /home/hice1/ayoon37/scratch/Training
10 python Test.py

sbatch submit.sbatch to submit

## 3.3.3, Storage

- mkdir Storage:
- mv example.txt /Storage
- cd Storage
- nano Example.py

## 3.3.4, Delete

1 cd /home/hice1/ayoon37/scratch/Storage
2 rm *.py

# 5.4

SSH and Environment Setup

1 ssh ayoon37@login-ice.pace.gatech.edu
2 module load anaconda3
3 conda create -name vip5
4 conda activate vip5
5 conda install -c conda-forge ase
6 conda install -c conda-forge sparc-x
7 cd scratch

Slurm Submission Bash Code

1 #!/bin/bash
2 # SBATCH --job-name=test_job
3 # SBATCH --output=test.out
4 # SBATCH --error=test.err
5 # SBATCH --nodes=1
6 # SBATCH --ntasks=1
7 # SBATCH --cpus-per-task=1
8 # SBATCH --time=5:00
9 
10 cd /home/hice1/ayoon37/scratch
11 python code.py

## 5.4.1, PBE

1 from sparc import SPARC
2 from ase.build import molecule
3 from ase.units import Bohr, Hartree, mol, kcal, kJ, eV
4 
5 # make the atoms
6 atoms = molecule('XXX')
7 atoms.cell = [[8,0,0],[0,8,0],[0,0,8]]
8 atoms.center()
9 
10 # setup calculator
11 parameters = dict(
12     EXCHANGE_CORRELATION='GGA_PBE',
13     D3_FLAG=1, # Grimme D3 dispersion correction
14     SPIN_TYP=0, # non spin-polarized calculation
15     KPOINT_GRID=[1,1,1], # molecule needs single kpt!
16     ECUT=500/Hartree, # set ECUT (Hartree) or h (Angstrom)
17     # h = 0.15,
18     TOL_SCF=1e-5,
19     RELAX_FLAG=1, # Do structural relaxation (only atomic positions)
20     PRINT_FORCES=1,
21     PRINT_RELAXOUT=1)
22 
23 calc = SPARC(atoms=atoms, **parameters)
24 
25 # set the calculator on the atoms and run
26 atoms.set_calculator(calc)
27 print(atoms.get_potential_energy())

Replace the "XXX" with H2O, CO2, and NH3

CO2 H2O NH3
Free Energy per atom -1.3026422117E+01 (Ha/atom) -5.9091464488E+00 (Ha/atom) -3.0423820785E+00 (Ha/atom)
Total free energy -3.9079266350E+01 (Ha) -1.7727439346E+01 (Ha) -1.2169528314E+01
(Ha)
Exchange correlation energy -1.0346416024E+01 (Ha) -4.8723761321E+00 (Ha) -4.1181041434E+00 (Ha)
Self and correction energy -5.8356705267E+01 (Ha) -2.6885066152E+01 (Ha) -2.0889501184E+01 (Ha)

CO2 -4.1280012214
H2O -2.7126289694  
NH3 -0.8158049333000003

CO2 H2O NH3
Total Time 68.228s 69.386s 141.823s
Calculation Time 0.050s 0.052s 0.034s

Comparison of my results vs experimentation:
- The variance between my results and experimentation seem to be quite large (when considering
percentile off), however since the scale is so small, these results/approximations could possibly
be considered to be fairly accurate.

Comparison between DFT Methods (B3LYP vs. PBE) I could not resolve this functional. I couldn't find the function in the documentation ([SPARC-X Documentation](https://github.com/SPARC-X/SPARC-X-API/blob/master/sparc/calculator.py)) and when asked on slack, the answer proved inconclusive.
The following comparison/results are from what I found online/independent research.

- Accuracy: B3LYP is a hybrid functional and generally offers better accuracy for various properties compared to PBE, which is a generalized gradient approximation (GGA) functional. B3LYP tends to provide more accurate energies but at a higher computational cost.

- Computational Time: B3LYP calculations typically take longer than PBE due to the increased complexity of the functional. B3LYP involves a mix of exact Hartree-Fock exchange with DFT, leading to higher computational demand.

## 5.4.2, CaLTA

W/O SPIN & D3 DISPERSION CORRECTION:

```python
from sparc import SPARC
from ase.build import molecule  
from ase.units import Bohr, Hartree, mol, kcal, kJ, eV
from ase.io import read, write

# make the atoms
atoms = molecule('CaLTA.vasp')
atoms.cell = [[8,0,0],[0,8,0],[0,0,8]]
atoms.center()

# setup calculator
parameters = dict(
    EXCHANGE_CORRELATION='GGA_PBE',
    D3_FLAG=0, # Grimme D3 dispersion correction
    SPIN_TYP=0, # non spin-polarized calculation
    KPOINT_GRID=[1,1,1], # molecule needs single kpt!
    ECUT=500/Hartree, # set ECUT (Hartree) or h (Angstrom)
    # h = 0.15,
    TOL_SCF=1e-5,
    RELAX_FLAG=1, # Do structural relaxation (only atomic positions)
    PRINT_FORCES=1,
    PRINT_RELAXOUT=1)

calc = SPARC(atoms=atoms, **parameters)

# set the calculator on the atoms and run
atoms.set_calculator(calc)
print(atoms.get_potential_energy() / eV)

=True)
27     atoms_copy.set_scaled_positions(scaled_positions)
28     
29     energy = atoms_copy.get_potential_energy()
30     
31     volumes.append(atoms_copy.get_volume())
32     energies.append(energy)
33 
34 plt.figure(figsize=(8, 6))
35 plt.plot(volumes, energies, marker='o', linestyle='-')
36 plt.xlabel('Cell Volume')
37 plt.ylabel('Potential Energy (eV)')
38 plt.title('Energy vs. Cell Volume')
39 plt.grid(True)
40 plt.show()

CUBIC EQUATION

1 import numpy as np
2 import matplotlib.pyplot as plt
3 from scipy.optimize import minimize
4 
5 volumes = np.array(volumes)
6 energies = np.array(energies)
7 
8 coefficients = np.polyfit(volumes, energies, 3) # Fit a polynomial of order 3 (cubic)
9 
10 def eos_function(volume):
11     return np.polyval(coefficients, volume)
12 
13 result = minimize(eos_function, volumes.mean(), method='Nelder-Mead')
14 
15 optimal_volume = result.x[0]
16 minimized_energy = result.fun
17 
18 plt.figure(figsize=(8, 6))
19 plt.plot(volumes, energies, marker='o', linestyle='-', label='Energy vs. Volume')
20 plt.plot(optimal_volume, minimized_energy, marker='o', markersize=8, color='red', label='Minimum Energy')
21 plt.xlabel('Cell Volume')
22 plt.ylabel('Potential Energy (eV)')
23 plt.title('Energy vs. Cell Volume with Fitted EOS')
24 plt.legend()
25 plt.grid(True)
26 plt.show()

# 6.4

## 6.4.1, Structural Relaxation

BUILD

1 from ase import Atoms
2 from ase.build import molecule
3 
4 CO2 = molecule('CO2')
5 CO2.set_distance(0, 1, distance=1.8)
6 print(CO2.get_positions())

SPARC

1 from sparc import SPARC
2 from ase.build import molecule
3 from ase.units import Bohr, Hartree, mol, kcal, kJ, eV
4 
5 # make the atoms
6 atoms = molecule('CO2')
7 atoms.set_distance(0, 1, distance=1.8)
8 atoms.cell = [[8,0,0],[0,8,0],[0,0,8]]
9 atoms.center()
10 
11 # setup calculator
12 parameters = dict(
13     EXCHANGE_CORRELATION='GGA_PBE',
14     D3_FLAG=0, # Grimme D3 dispersion correction
15     SPIN_TYP=0, # non spin-polarized calculation
16     KPOINT_GRID=[1,1,1], # molecule needs single kpt!
17     ECUT=500/Hartree, # set ECUT (Hartree) or h (Angstrom)
18     # h = 0.15,
19     TOL_SCF=1e-5,
20     RELAX_FLAG=1, # Do structural relaxation (only atomic positions)
21     PRINT_FORCES=1,
22     PRINT_RELAXOUT=1)
23 
24 calc = SPARC(atoms=atoms, **parameters)
25 
26 # set the calculator on the atoms and run
27 atoms.set_calculator(calc)
28 print(atoms.get_potential_energy())

QUANTUM ESPRESSO

1 image = molecule('CO2', vacuum=10.)
2 image.set_cell([10, 10, 10])
3 image.set_pbc([1,1,1])
4 image.center()
5 image.rattle(0.0001)
6 calc = Espresso(atoms=image,
7                 pw=500.0,
8                 xc='PBE',
9                 kpts="gamma")
10 
11 dyn = BFGS(image, logfile='opt_pbe.log', trajectory='h2o_pbe_optimization.traj')
12 dyn.run(fmax=0.005)
13 image.calc.close()
14 
15 print(image.get_potential_energy())

## 6.4.3, CO2 Adsorption

RELAX PRISTINE MOF

1 import os
2 from ase import Atoms, io
3 from ase.io import read, write
4 from ase.build import bulk, molecule, surface, add_adsorbate
5 from ase.units import Bohr, Hartree, mol, kcal, kJ, eV
6 from ase.constraints import FixAtoms
7 from sparc import SPARC
8 
9 parameters = dict(
10     EXCHANGE_CORRELATION='GGA_PBE',
11     D3_FLAG=1, # Grimme D3 dispersion correction
12     SPIN_TYP=0, # spin-polarized calculation
13     KPOINT_GRID=[1,1,1],
14     ECUT=600/Hartree, # set ECUT (Hartree) or h (Angstrom)
15     # h = 0.15,
16     TOL_SCF=1e-4,
17     RELAX_FLAG=1,
18     TOL_RELAX=2.00E-03, # convergence criteria (maximum force) (Ha/Bohr)
19     PRINT_FORCES=1,
20     PRINT_RELAXOUT=1)
21 
22 cwd = os.getcwd()
23 parameters['directory'] = cwd + '/Exercise_3/pristine'
24 
25 atoms = read('CaLTA.vasp')
26 c = FixAtoms(indices=[atom.index for atom in atoms])
27 atoms.set_constraint(c)
28 
29 calc = SPARC(atoms=atoms, **parameters)
30 atoms.set_calculator(calc)
31 
32 eng_pristine = atoms.get_potential_energy()
33 atoms.write('Exercise_3/CONTCAR_3_pristine')
34 eng_pristine

SINGLE SITE

1 atoms = read('CaLTA.vasp')
2 c = FixAtoms(indices=[atom.index for atom in atoms])
3 atoms.set_constraint(c)
4 
5 CO2 = CO2_original.copy()
6 CO2.cell = atoms.cell
7 CO2.rotate(-45, 'y', 'COM')
8 d_trans = atoms[73].position - CO2[0].position + np.array([1,1,2])
9 CO2.translate(d_trans)
10 atoms.extend(CO2)
11 
12 atoms.write('Exercise_3/POSCAR_3_SS')?

DUAL SITE  

1 atoms = read('CaLTA.vasp')
2 c = FixAtoms(indices=[atom.index for atom in atoms])
3 atoms.set_constraint(c)
4 
5 CO2 = CO2_original.copy()
6 CO2.cell = atoms.cell
7 d_trans = atoms[73].position - CO2[1].position + np.array([3.55,1,0])
8 CO2.translate(d_trans)
9 atoms.extend(CO2)
10 
11 atoms.write('Exercise_3/POSCAR_3_DS')

RESULTS
Pristine: -0.203
Single Site: -0.767
Dual Site: -0.470

CONCLUSION:
Seeing as the greater adsorption energy equates to a more stable active site - it should also be noted that we are taking the absolute value of the results, for the negative simply signifies attraction/adsorption. Furthermore there is a clear "peak" in
