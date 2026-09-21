# REBAS

REBAS is a replica exchange method intended to enhance the efficiency  

## Commits used

Robosample commit: 9b1a35012a0bda3e6c17540e8fc63fe087d258bb  
Molmodel commit: f9e4dfd520451189b616159dcae43e2329e4dccd singularity  
Simbody commit: 749f47ef9dfaef7e40594a917ee16d2ce4e9dbc8 master  
OpenMM commit: 63e113d9557199d36587457deb159ae34fd75188 drilling  

## Simulations seeds
Seed digits codification:
- first two digits are reserved for molecule
- third digit is for type of simulation which is <0:Default> <1:REMC> <2:RENEMC> or <3:REBAS>
- fourth digit is for eventual subtypes
- fifth digit is for Gibbs identifiers
- last two digits are for indicating the repeat number

## Simulation locations
here /home/laurentiu/0Work/robo/tfep/prod/ethane/examSlider/ seeds: [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1010200, 1010201, 1010202, 1031200, 1031201, 1031202, 1032200, 1032201, 1032202, 1033200, 1033201, 1033202, 1034200, 1034201, 1034202, 1034204, 1035200, 1035202, 1035204, 1036200, 1036201, 1036202, 1037200, 1037201, 1037202]
albicastro 49 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [2010500, 2010502, 2010502, 2010503, 2030500, 2030502, 2030502, 2030503]
salieri 57 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [3030500, 3030501]
salieri 57 /home/laurentiu/0Work/robo/tfep/prod/adk/everyRepl.00 seeds: [4010500, 4010501, 4030501]
handel 54 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/  seeds: [3010504, 3010504]
corelli 47 0Work/robo/tfep/prod/trpch/everyRepl.09.mi6.04/ seeds: [3030504, 3030504]
corelli 47 0Work/robo/tfep/prod/adk/everyRepl.00/ seeds: [4010502, 4010503, 4030502, 4030503]

## Commands

Commands used to generate paper figures

### ======== SCALING math

python /home/laurentiu/git6/REBAS/scaleMath/00.py --mean 0 --std 3 --sampleSize 10000 --scaleFactor 3 > scaleMath/x.txt

### ========= PRELIMINRY Harmonic oscillator

python /home/laurentiu/git6/REBAS/harmOscSims/experiment.py --nofExperiments 1 --worlds 0 1 --sampleSize 10 --coldTargetMean 1.0 --coldTargetStd 3
python ~/scripts/quick_scatter_gpt.py --dir ./ --inFNRoot pythonStudy/harmonic.RENS.wo2.out --inCols 0 1 --histogram --nbins 100 --merge --skipheadrows 1

### ========= DRILLING  

Commits: 

python simulate.rebas.py --name 2but --top 2but/ligand.prmtop --rst7 2but/ligand.rst7 --rstDir rest.2but.00/1111/ --equilSteps 0 --prodSteps 10000 --writeFreq 1 --baseTemperature 300.00 --baseTdiff 300 --nofReplicas 2 --runType REMC --seed 1111 --flexFNs 2but/ligand.td.flex 2but/ligand.td.flex --FixmanTorque False False False --roll False False False --printDrlEnergies True --drlPrintStride 1 > out.2but.out
egrep "drl World|REX, ., ., 0" out.2but.out | sed 's/,//g' | awk 'BEGIN{nofAtoms=15} $1=="REX"{printf("%s ", $5)} $1=="drl"{printf("%s %s ", $3, $4)} $3=="couE_sum"{ printf("\n")} END{printf("\n")}' | grep -v "^bon" > ene.2but.ene
python readAndReshape.py --input ene.2but.ene --output ene.2but.reshaped
python ~/git6/REBAS/rebas.py --dir ./ --inFNRoots ene.2but.reshaped --drill all --moleculeName 2but --trajBurnin 0


python simulate.rebas.py --name ala1 --top ala1/ligand.prmtop --rst7 ala1/ligand.rst7 --rstDir rest.ala1.00/2010510/ --equilSteps 0 --p^CdSteps 10000 --writeFreq 1 --baseTemperature 1000.00 --baseTdiff 200 --nofReplicas 2 --runType REMC --seed 2010510 --flexFNs ala1/ligand.flex.td ala1/ligand.flex.td --FixmanTorque False False False --roll False False False --printDrlEnergies True --drlPrintStride 1 > out.2010510 &
egrep "drl World|REX, ., ., 0" out.ala1.out | sed 's/,//g' | awk 'BEGIN{nofAtoms=22} $1=="REX"{printf("%s ", $5)} $1=="drl"{printf("%s ", $0)} $3=="couE_sum"{ printf("\n")} END{printf("\n")}' | grep -v "^bon" | grep "^1000 " > ene.ala1.enextend.1000

Get distances:
cd drill/
vmd -f 2but/ligand.prmtop 2but_1111.repl0.dcd -e measure.2but.tcl
python ~/git6/REBAS/rebas.py --dir ./ --inFNRoots out.2but.dist --drill all --moleculeName 2but --trajBurnin 0

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
