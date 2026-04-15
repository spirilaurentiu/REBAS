# ======== VALIDATION probability distributions

python ~/git6/REBAS/prod_valid_eth_thermo.py --dir prod/ethane/examSlider/ --inFNRoots out.proc.T300.10000 out.proc.T300.101 out.proc.T300.1033 out.proc.T600.10000 out.proc.T600.101 out.proc.T600.1033 --nbins 100 --Ts 300 300 300 600 600 600

# ======== VALIDATION dpe

python ~/git6/REBAS/rebas.py --dir prod/ala1/everyRepl.00.mi6/ --inFNRoots out.detail.2030500 --cols replicaIx thermoIx wIx acc pe_o pe_n JDetLog --filterBy wIx=3 thermoIx=1,2 --checks dpe
python ~/git6/REBAS/rebas.py --dir prod/trpch/everyRepl.09.mi6/ --inFNRoots out.detail.3030500 --cols replicaIx thermoIx wIx acc pe_o pe_n JDetLog --filterBy wIx=3 thermoIx=1,2 --checks dpe


# ======== EFFICIENCY




# ======== ANALYSIS
