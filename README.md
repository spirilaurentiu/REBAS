# REBAS

REBAS is a replica exchange method intended to enhance the efficiency  

# Commits used

Robosample commit: 9b1a35012a0bda3e6c17540e8fc63fe087d258bb  
Molmodel commit: f9e4dfd520451189b616159dcae43e2329e4dccd singularity  
Simbody commit: 749f47ef9dfaef7e40594a917ee16d2ce4e9dbc8 master  
OpenMM commit: 63e113d9557199d36587457deb159ae34fd75188 drilling  

# Simulations seeds
Seed digits codification:
 - first two digits are reserved for molecule
 - third digit is for type of simulation which is <0:Default> <1:REMC> <2:RENEMC> or <3:REBAS>
 - fourth digit is for eventual subtypes
 - fifth digit is for Gibbs identifiers
 - last two digits are for indicating the repeat number

# Simulation locations
here /home/laurentiu/0Work/robo/tfep/prod/ethane/examSlider/ seeds: [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1010200, 1010201, 1010202, 1031200, 1031201, 1031202, 1032200, 1032201, 1032202, 1033200, 1033201, 1033202, 1034200, 1034201, 1034202, 1034204, 1035200, 1035202, 1035204, 1036200, 1036201, 1036202, 1037200, 1037201, 1037202]
albicastro 49 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [2010500, 2010502, 2010502, 2010503, 2030500, 2030502, 2030502, 2030503]
salieri 57 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [3030500, 3030501]
handel 54 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/  seeds: [3010504, 3010504]
corelli 47 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [3030504, 3030504]

# Commands

### ======== PROCESS data

python ~/git6/REBAS/processOuts.py --dir prod/ethane/examSlider/ --inFNRoots out --procSuffix detail

### ======== VALIDATION probability distributions

python ~/git6/REBAS/prod_valid_eth_thermo.py --dir prod/ethane/examSlider/ --inFNRoots out.proc.T300.10000 out.proc.T300.101 out.proc.T300.1033 out.proc.T600.10000 out.proc.T600.101 out.proc.T600.1033 --nbins 100 --Ts 300 300 300 600 600 600

### ======== VALIDATION dpe

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi7/ --inFNRoots out.detail.2030500 --cols replicaIx thermoIx wIx acc pe_o pe_n JDetLog --filterBy wIx=3 thermoIx=1,2,3,4,5,6,7,8 --checks dpe  

### ======== EFFICIENCY thermodynamic space

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots out.detail --cols replicaIx thermoIx wIx acc pe_o --filterBy wIx=0 replicaIx=0,1,2,3,4,5,6,7,8,9,10,11,12,13 --figures rex_eff  
ython ~/git6/REBAS/rebas.py --dir prod/trpch/everyRepl.09.mi6/final/ --inFNRoots out.detail --cols replicaIx thermoIx wIx acc pe_o --filterBy wIx=0 replicaIx=0,1,2,3,4,5,6,7,8,9,10,11,12,13 --figures rex_eff  

### ======== EFFICIENCY conformational space

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots ala1_ --topology ala1/ligand.prmtop --cols replicaIx thermoIx wIx acc pe_o --filterBy thermoIx=0 --figures traj_stats  
python ~/git6/REBAS/rebas.py --dir prod/trpch/everyRepl.09.mi6.04/ --inFNRoots trpch_ --topology trpch/ligand.prmtop --cols replicaIx thermoIx wIx acc pe_o --filterBy thermoIx=0,13 --figures traj_stats
python ~/git6/REBAS/rebas.py --moleculeName trpch --dir prod/trpch/everyRepl.09.mi6.04/ --inFNRoots trpch_ --topology trpch/ligand.prmtop --cols replicaIx thermoIx wIx acc pe_o --filterBy thermoIx=0,1 --figures traj_stats --trajBurnin 10000

python ~/git6/REBAS/rebas.py --moleculeName trpch --dir prod/trpch/everyRepl.09.mi6.04/ --inFNRoots trpch_ --topology trpch/ligand.prmtop --cols replicaIx thermoIx wIx acc pe_o --filterBy thermoIx=0 --figures circ_ACF --trajBurnin 30000 --batDihIxs 1 --acfKinds dihedrals --acfMaxLag 200 --acfBatchSize 16


### ======== ANALYSIS
Not yet
