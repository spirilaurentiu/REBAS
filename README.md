# REBAS

REBAS is a replica exchange method intended to enhance the efficiency  

## Reproduce

### Commits used

Robosample commit: 9b1a35012a0bda3e6c17540e8fc63fe087d258bb  
Molmodel commit: f9e4dfd520451189b616159dcae43e2329e4dccd singularity  
Simbody commit: 749f47ef9dfaef7e40594a917ee16d2ce4e9dbc8 master  
OpenMM commit: 63e113d9557199d36587457deb159ae34fd75188 drilling  

### ======== PROCESS data

python ~/git6/REBAS/processOuts.py --dir prod/ethane/examSlider/ --inFNRoots out --procSuffix detail

### ======== VALIDATION probability distributions

python ~/git6/REBAS/prod_valid_eth_thermo.py --dir prod/ethane/examSlider/ --inFNRoots out.proc.T300.10000 out.proc.T300.101 out.proc.T300.1033 out.proc.T600.10000 out.proc.T600.101 out.proc.T600.1033 --nbins 100 --Ts 300 300 300 600 600 600

### ======== VALIDATION dpe

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots out.detail.2030500 --cols replicaIx thermoIx wIx acc pe_o pe_n JDetLog --filterBy wIx=3 thermoIx=1,2 --checks dpe  
python ~/git6/REBAS/rebas.py --dir prod/trpch/everyRepl.09.mi6/ --inFNRoots out.detail.3030500 --cols replicaIx thermoIx wIx acc pe_o pe_n JDetLog --filterBy wIx=3 thermoIx=1,2 --checks dpe  

### ======== EFFICIENCY thermodynamic space

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots out.detail --cols replicaIx thermoIx wIx acc pe_o --filterBy wIx=0 replicaIx=0,1,2,3,4,5,6,7,8,9,10,11,12,13 --figures rex_eff  
ython ~/git6/REBAS/rebas.py --dir prod/trpch/everyRepl.09.mi6/final/ --inFNRoots out.detail --cols replicaIx thermoIx wIx acc pe_o --filterBy wIx=0 replicaIx=0,1,2,3,4,5,6,7,8,9,10,11,12,13 --figures rex_eff  

### ======== EFFICIENCY conformational space

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots ala1_ --topology ala1/ligand.prmtop --cols replicaIx thermoIx wIx acc pe_o --filterBy thermoIx=0 --figures traj_stats  

### ======== ANALYSIS

python  
