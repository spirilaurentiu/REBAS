
#include 00.h

using namespace SimTK;

class Context{
public:
    void Initialize();
    bool attemptREXSwap(int thermoState_C, int thermoState_H);
    void RunREX(int equilRounds, int prodRounds);
};

class HMCSampler{
public:
    bool sample_iteration();
    SimTK::Real getDistortJacobianDetLog();
    void setDistortJacobianDetLog(SimTK::Real val);
    bool reinitialize(const SimTK::State& state, std::stringstream& worldOutStream, bool verbose);
    int getDistortOption();
    bool getAcc();
    SimTK::Real fix_set;
};

void Context::Initialize() {

    /*
	worlds[0].setDuMMAtomIndexes(); // REVISE

	// setMyContext
	for (int worldIx = 0; worldIx < worlds.size(); worldIx++) {
		worlds[worldIx].setMyContext(this);
	}

	// Initialize the Z matrix
	const auto& firstWorldsAtomsLocations = worlds[firstWIx].getAtomsLocationsInGround(worlds[0].integ->updAdvancedState());

	// Get Z-matrix indexes table
    # pragma region ZMATRIX
	calcZMatrixTable();
	PrintZMatrixTable();
	reallocZMatrixBAT();
	calcZMatrixBAT(firstWIx, firstWorldsAtomsLocations);
	PrintZMatrixBAT();
	PrintZMatrixMobods(firstWIx, lastAdvancedState);
	
	for(int k = 0; k < nofReplicas; k++){
		replicas[k].reallocZMatrixBAT();
		replicas[k].calcZMatrixBAT(firstWorldsAtomsLocations);
	}
	
	// They all start with replica 0 coordinates
	for (int worldIx = 0; worldIx < worlds.size(); worldIx++) {
		World& world = worlds[worldIx];
		addSubZMatrixBATsToWorld(worldIx, 0);
	}
    # pragma endregion ZMATRIX

	// Consider renaming
	loadReplica2ThermoIxs();
	PrintReplicas();
    */

	// @@@@@@@@@@@@ Set QScaleFactors to sqrt(Ti/Tj)
	PrepareNonEquilibriumParams_Q(){
    Context::PrepareNonEquilibriumParams_Q(){

        // Set the even scale factors equal to the sqrt(Ti/Tj)
        for(size_t thermoIx = 0; thermoIx < nofThermodynamicStates - 1; thermoIx += 2){
            // s_i = T_j
            qScaleFactorsEven.at(thermoIx)     = thermodynamicStates[thermoIx + 1].getTemperature();
            qScaleFactorsEven.at(thermoIx + 1) = thermodynamicStates[thermoIx].getTemperature();

            // s_i /= T_i
            qScaleFactorsEven.at(thermoIx)     /= thermodynamicStates[thermoIx].getTemperature();
            qScaleFactorsEven.at(thermoIx + 1) /= thermodynamicStates[thermoIx + 1].getTemperature();

            // s_i = sqrt(s_i)
            qScaleFactorsEven.at(thermoIx) = std::sqrt(qScaleFactorsEven.at(thermoIx));
            qScaleFactorsEven.at(thermoIx + 1) = std::sqrt(qScaleFactorsEven.at(thermoIx + 1));
        }

        // Set the odd scale factors equal to the sqrt(Ti/Tj)
        for(size_t thermoIx = 1; thermoIx < nofThermodynamicStates - 1; thermoIx += 2){

            // s_i = T_j
            qScaleFactorsOdd.at(thermoIx)     = thermodynamicStates[thermoIx + 1].getTemperature();
            qScaleFactorsOdd.at(thermoIx + 1) = thermodynamicStates[thermoIx].getTemperature();

            // s_i /= T_i
            qScaleFactorsOdd.at(thermoIx)     /= thermodynamicStates[thermoIx].getTemperature();
            qScaleFactorsOdd.at(thermoIx + 1) /= thermodynamicStates[thermoIx + 1].getTemperature();

            // s_i = sqrt(s_i)
            qScaleFactorsOdd.at(thermoIx) = std::sqrt(qScaleFactorsOdd.at(thermoIx));
            qScaleFactorsOdd.at(thermoIx + 1) = std::sqrt(qScaleFactorsOdd.at(thermoIx + 1));
        }

    } // _end_ PrepareNonEquilibriumParams_Q
    } // _end_ PrepareNonEquilibriumParams_Q

}


/*! <!-- Attempt swap between replicas r_i and r_j
 * Code inspired from OpenmmTools
 * Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations
 * as Gibbs multistate: Simple improvements for enhanced mixing. J. Chem. Phys.
 * , 135:194110, 2011. DOI:10.1063/1.3660669
 *  replica_i and replica_j are variable
 * --> */
bool Context::attemptREXSwap(int thermoState_C, int thermoState_H)
{
	bool returnValue = false;

	// Extract information only from Replica and ThermodynamicState objects
	// Do not use World objects
	#pragma region CONVENIENT_VARS

	// Get replicas' thermodynamic states indexes
	//int thermoState_C = replica2ThermoIxs[replica_X];
	//int thermoState_H = replica2ThermoIxs[replica_Y];

	int replica_X = thermo2ReplicaIxs[thermoState_C];
	int replica_Y = thermo2ReplicaIxs[thermoState_H];
	
	// Record this attempt
	nofAttemptedSwapsMatrix[thermoState_C][thermoState_H] += 1;
	nofAttemptedSwapsMatrix[thermoState_H][thermoState_C] += 1;

	// For useful functions
	auto genericSampler = worlds[0].updSampler(0);

	// ----------------------------------------------------------------
	// Convenient vars (Ballard-Jarzinski nomenclature)
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real beta_C = thermodynamicStates[thermoState_C].getBeta();
	SimTK::Real beta_H = thermodynamicStates[thermoState_H].getBeta();
	
	SimTK::Real U_X0 = replicas[replica_X].getPotentialEnergy(); // last equil potential
	SimTK::Real U_Y0 = replicas[replica_Y].getPotentialEnergy(); // last equil potential

	SimTK::Real refU_X0 = replicas[replica_X].getReferencePotentialEnergy(); // last equil reference potential
	SimTK::Real refU_Y0 = replicas[replica_Y].getReferencePotentialEnergy(); // last equil reference potential

	SimTK::Real W_X = replicas[replica_X].getWORK(); // work without Jacobian
	SimTK::Real W_Y = replicas[replica_Y].getWORK(); // work without Jacobian

	SimTK::Real U_Xtau = replicas[replica_X].get_WORK_PotentialEnergy_New(); // last non-equil potential
	SimTK::Real U_Ytau = replicas[replica_Y].get_WORK_PotentialEnergy_New(); // last non-equil potential

	SimTK::Real refU_Xtau = replicas[replica_X].get_WORK_ReferencePotentialEnergy_New(); // last non-equil potential
	SimTK::Real refU_Ytau = replicas[replica_Y].get_WORK_ReferencePotentialEnergy_New(); // last non-equil potential

	SimTK::Real lnJac_X = replicas[replica_X].get_WORK_Jacobian(); // non-equil Jacobian
	SimTK::Real lnJac_Y = replicas[replica_Y].get_WORK_Jacobian(); // non-equil Jacobian


	// ----------------------------------------------------------------
	// Reduced potentials X0
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real uC_X0 = beta_C * U_X0; // Replica i reduced potential in state i
	SimTK::Real uH_Y0 = beta_H * U_Y0; // Replica j reduced potential in state j
	SimTK::Real uH_X0 = beta_H * U_X0; // Replica i reduced potential in state j
	SimTK::Real uC_Y0 = beta_C * U_Y0; // Replica j reduced potential in state i

	SimTK::Real ref_uC_X0 = beta_C * refU_X0; // Replica i reduced reference potential in state i
	SimTK::Real ref_uH_Y0 = beta_H * refU_Y0; // Replica j reduced reference potential in state j
	SimTK::Real ref_uH_X0 = beta_H * refU_X0; // Replica i reduced reference potential in state j
	SimTK::Real ref_uC_Y0 = beta_C * refU_Y0; // Replica j reduced reference potential in state i

	// ----------------------------------------------------------------
	// Reduced potential Xtau
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real uC_Xtau = beta_C * U_Xtau; // Replica i reduced potential in state i
	SimTK::Real uH_Ytau = beta_H * U_Ytau; // Replica j reduced potential in state j
	SimTK::Real uH_Xtau = beta_H * U_Xtau; // Replica i reduced potential in state j
	SimTK::Real uC_Ytau = beta_C * U_Ytau; // Replica j reduced potential in state i

	SimTK::Real ref_uC_Xtau = beta_C * refU_Xtau; // Replica i reduced reference potential in state i
	SimTK::Real ref_uH_Ytau = beta_H * refU_Ytau; // Replica j reduced reference potential in state j
	SimTK::Real ref_uH_Xtau = beta_H * refU_Xtau; // Replica i reduced reference potential in state j
	SimTK::Real ref_uC_Ytau = beta_C * refU_Ytau; // Replica j reduced reference potential in state i

	#pragma endregion CONVENIENT_VARS

	// ----------------------------------------------------------------
	// LOGP ENERGY EQUILIBRIUM
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real ETerm_equil    = ref_uH_X0 - ref_uC_X0;
				ETerm_equil   += ref_uC_Y0 - ref_uH_Y0;
				ETerm_equil = -1.0 * ETerm_equil;

	// ----------------------------------------------------------------
	// LOGP ENERGY NON-EQUILIBRIUM
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real ETerm_nonequil  = ref_uH_Xtau - ref_uC_Xtau;
			    ETerm_nonequil += ref_uC_Ytau - ref_uH_Ytau;
				ETerm_nonequil = -1.0 * ETerm_nonequil;

	// ----------------------------------------------------------------
	// LOGP WORK
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// Get work from X replica
	SimTK::Real Work_X = (ref_uH_Xtau - ref_uC_X0) - lnJac_X;                           // variant 2

	// Get work from Y replica
	SimTK::Real Work_Y = (ref_uC_Ytau - ref_uH_Y0) - lnJac_Y;                           // variant 2

	// Get total work
	SimTK::Real WTerm = -1.0 * (Work_X + Work_Y);

	// ----------------------------------------------------------------
	// CORRECTION TERM FOR REBAS : probability of choosing
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	#pragma region CORRECTION_TERM

	SimTK::Real correctionTerm = 1.0;
	SimTK::Real miu_C = qScaleFactorsMiu.at(thermoState_C);
	SimTK::Real miu_H = qScaleFactorsMiu.at(thermoState_H);
	SimTK::Real std_C = qScaleFactorsStd.at(thermoState_C);
	SimTK::Real std_H = qScaleFactorsStd.at(thermoState_H);
	
	SimTK::Real s_X = qScaleFactors.at(thermoState_C);
	SimTK::Real s_Y = qScaleFactors.at(thermoState_H);
	SimTK::Real s_X_1 = 1.0 / s_X;
	SimTK::Real s_Y_1 = 1.0 / s_Y;

	// Correction term is 1 for now
	SimTK::Real qC_s_X = 1.0, qH_s_Y = 1.0, qH_s_X_1 = 1.0, qC_s_Y_1 = 1.0;
	correctionTerm = (qH_s_X_1 * qC_s_Y_1) / (qC_s_X * qH_s_Y);

	#pragma endregion CORRECTION_TERM


	// ----------------------------------------------------------------
	// EVALUATE
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	SimTK::Real log_p_accept = 1.0;

	// Calculate log_p_accept
	if(getRunType() == RUN_TYPE::REMC){

		log_p_accept = ETerm_equil ;

	}else if(getRunType() == RUN_TYPE::RENEMC){

		log_p_accept = ETerm_nonequil + std::log(correctionTerm) ;

	}else if( getRunType() == RUN_TYPE::RENE){

		log_p_accept = WTerm + std::log(correctionTerm) ;

	}

	// Draw from uniform distribution
	SimTK::Real unifSample = uniformRealDistribution(randomEngine);

	bool testingMode = false; 
	// Accept
	if((log_p_accept >= 0.0) || (unifSample < std::exp(log_p_accept))){

		if( (getRunType() == RUN_TYPE::RENE) || (getRunType() == RUN_TYPE::REBASONTOP)){

			replicas[replica_X].incrementWorldsNofSamples();
			replicas[replica_Y].incrementWorldsNofSamples();
			thermodynamicStates[thermoState_C].incrementWorldsNofSamples();
			thermodynamicStates[thermoState_H].incrementWorldsNofSamples();

			bool onlyNonequilWorlds = true;
			for(int wIx = 0; wIx < nofWorlds; wIx++){
				if(worlds[wIx].getSampler(0)->getDistortOption() == 0){
					onlyNonequilWorlds = false;
					break;
				}
			}

			if(onlyNonequilWorlds){
				replicas[replica_X].incrementNofSamples();
				replicas[replica_Y].incrementNofSamples();
				thermodynamicStates[thermoState_C].incrementNofSamples();
				thermodynamicStates[thermoState_H].incrementNofSamples();
			}
			

		}

		if((getRunType() == RUN_TYPE::RENE) || (getRunType() == RUN_TYPE::RENEMC) || (getRunType() == RUN_TYPE::REBASONTOP)){
			
			// Update replicas coordinates from work generated coordinates
			set_WORK_CoordinatesAsFinal(replica_X);
			set_WORK_CoordinatesAsFinal(replica_Y);

			// Update replica's energy from work last potential energy
			set_WORK_PotentialAsFinal(replica_X);
			set_WORK_PotentialAsFinal(replica_Y);
		}

		// Swap thermodynamic states
		swapThermodynamicStates(replica_X, replica_Y);
		swapPotentialEnergies(replica_X, replica_Y);
		swapReferencePotentialEnergies(replica_X, replica_Y);

		std::cout << "1" 
		<<", " << unifSample
		<< endl << endl;

		returnValue = true;

	// Reject
	}else{

		rewindReplica();

		// Return to equilibrium worlds coordinates
		// - no need because it is restored in RunREX
		// Return to equilibrium worlds energies
		// - no need because it is restored in RunREX
		// Don't swap thermodynamics states nor energies

		std::cout << "0"
		<<", " << unifSample 
		<< endl << endl;

		returnValue = false;
	}

	return returnValue;

}



/*! <!-- HELPER(RunREX) Run a range of worlds for a replica --> */
void Context::runReplicaWorldRange(
		int replicaIx,
		int startWorldCnt, int nofWorldsCounted,
		bool isNonEquilibrium)
{

			for(int thWCnt = startWorldCnt; thWCnt < nofWorldsCounted; thWCnt++){

				if(thWCnt != startWorldCnt){ // don't transfer if it's the first world in the range
					if(thWCnt != 0){ // don't transfer if it's the first world in the range and it's also the first world overall
						transferCoordinates_WorldToWorld(thermoWorldIxs[thWCnt - 1], wIx);
						transferQStatistics(thermoIx, thermoWorldIxs[thWCnt - 1], wIx); // 
					}
				}				

				// Run
				bool validated = true;
				validated = RunWorld(wIx, string("REX, ") + to_string(replicaIx) + string(", ") + to_string(thermoIx) + ", " + to_string(wIx)) && validated;

				// Calculate Q statistics
				if(sampler_p->getAcc() == true){
					thermoState.calcQStats(wIx, currWorld.getBMps(), currWorld.getPFrs(), currWorld.getAdvancedQs(), currWorld.getNofSamples());
				}else{
					thermoState.calcQStats(wIx, currWorld.getBMps(), currWorld.getPFrs(), SimTK::Vector(currWorld.getNQs(), SimTK::Real(0)), currWorld.getNofSamples());
				}

				// ======================== EQUILIBRIUM ======================
				if(distortIx == 0){

					transferCoordinates_WorldToReplica(wIx, replicaIx);

					replica.setPotentialEnergy(currWorld.calcPotentialEnergy());
					replica.setFixman(sampler_p->fix_set);
					replica.setReferencePotentialEnergy(OMMRef_calcPotential(replica.getAtomsLocationsInGround(), true, true));
				} // __end__ Equilibrium 

				// ======================== NON-EQUILIBRIUM ======================
				else{ // (distortIx != 0)
					replica.updWORK() += currWorld.getWork();  // TODO merge with Jacobians
					replica.upd_WORK_Jacobian() += sampler_p->getDistortJacobianDetLog();

					transferCoordinates_WorldToReplica_WORK(wIx, replicaIx);

					replica.set_WORK_PotentialEnergy_New(currWorld.calcPotentialEnergy());
					replica.set_WORK_Fixman(sampler_p->fix_set);
					replica.set_WORK_ReferencePotentialEnergy_New(OMMRef_calcPotential(replica.get_WORK_AtomsLocationsInGround(), true, true));
				} // __end__ Non/Equilibrium

				// Increment the nof samples for replica and thermostate
				replica.incrementWorldsNofSamples(1);
				thermoState.incrementWorldsNofSamples(1);
			} // _end_ loop through worlds (EQUILIBRIUM)

}

/*!
 * <!-- Run replica exchange protocol -->
*/
void Context::RunREX(int equilRounds, int prodRounds)
{

	// Useful vars
	int nofMixes = requiredNofRounds;
	int currFrontWIx = -1;

	// @@@@@@@@@@ LOOP THROUGH ROUNDS --------------------------------------------------------->
	for(size_t mixi = 0; mixi < equilRounds + prodRounds; mixi++) {

		// Reset replica exchange pairs vector
		if(getRunType() != RUN_TYPE::DEFAULT){
			if(replicaMixingScheme == ReplicaMixingScheme::neighboring){
				setReplicaExchangePairs(mixi % 2);
			}
		}

		// Update work scale factors
		updThermostatesQScaleFactors(mixi);

		// @@@@@@@@@@ LOOP THROUGH REPLICAS (EQUILIBRIUM) ------------------------------------->
		for (int replicaIx = 0; replicaIx < nofReplicas; replicaIx++){ // BY_REPLICA

			# pragma region CONVENIENT_VARS_REPLICA
			int N_1_wCnt = -1, N_2_wCnt = -1;
            int nofEquilibriumWorlds = -1, nofNonequilibriumWorlds = -1;

			const Partitioning& thermoNonequilPart = thermoState.getNonequilPartitioning();

			N_1_wCnt = thermoNonequilPart.N1_wCnt;
			N_2_wCnt = thermoNonequilPart.N2_wCnt;
			nofEquilibriumWorlds = thermoNonequilPart.nofEquilibriumWorlds;
			nofNonequilibriumWorlds = thermoNonequilPart.nofNonequilibriumWorlds;
			# pragma endregion CONVENIENT_VARS_REPLICA

			// Update BAT map for all the replica's world
			updSubZMatrixBATsToAllWorlds(replicaIx);

			setReplicasWorldsParameters(replicaIx, false, true, mixi);

			transferCoordinates_ReplicaToWorld(replicaIx, 0);
			transferQStatistics(thermoIx, thermoWorldIxs.back(), 0);

			replica.updWORK() = 0.0;
			replica.upd_WORK_Jacobian() = 0.0;

			// @@@@@@@@@@ LOOP THROUGH EQUILIBRIUM WORLDS --------------------------------------------->
			runReplicaWorldRange(replicaIx, 0, nofEquilibriumWorlds, false);

			// Write log and DCD
			if(nofNonequilibriumWorlds == 0){
				writeReplicaLogAndDCD(mixi, replicaIx, printFreq);
			}

			replica.incrementNofSamples(1);
			thermoState.incrementNofSamples(1);

			SimTK::State& state = worlds[thermoWorldIxs[N_2_wCnt]].integ->updAdvancedState();
			replica.calcZMatrixBAT( worlds[thermoWorldIxs[N_2_wCnt]].getAtomsLocationsInGround( state ));

		} // _end_ loop through replicas (EQUILIBRIUM)



		if((getRunType() == RUN_TYPE::REBASONTOP) && (nofReplicas != 1)){
			setRunType(RUN_TYPE::REMC);
			mixReplicas(mixi);
			PrintNofAcceptedSwapsMatrix();
			//mixi++;
			setRunType(RUN_TYPE::REBASONTOP);
		}



		// @@@@@@@@@@ LOOP THROUGH REPLICAS (NON-EQUILIBRIUM) ------------------------------------->

		// Update work scale factors
		updThermostatesQScaleFactors(mixi);	

		for (int replicaIx = 0; replicaIx < nofReplicas; replicaIx++){ // BY_REPLICA

			# pragma region CONVENIENT_VARS_REPLICA

			int N_1_wCnt = -1, N_2_wCnt = -1;
			int nofEquilibriumWorlds = -1, nofNonequilibriumWorlds = -1;

			const Partitioning& thermoNonequilPart = thermoState.getNonequilPartitioning();

			N_1_wCnt = thermoNonequilPart.N1_wCnt;
			N_2_wCnt = thermoNonequilPart.N2_wCnt;
			nofEquilibriumWorlds = thermoNonequilPart.nofEquilibriumWorlds;
			nofNonequilibriumWorlds = thermoNonequilPart.nofNonequilibriumWorlds;
			# pragma endregion CONVENIENT_VARS_REPLICA

			// Update BAT map for all the replica's world
			updSubZMatrixBATsToAllWorlds(replicaIx);

			if(nofNonequilibriumWorlds){

				setReplicasWorldsParameters(replicaIx, false, true, mixi);

				transferCoordinates_ReplicaToWorld(replicaIx, thermoWorldIxs[N_1_wCnt]);
				transferQStatistics(thermoIx, thermoWorldIxs.back(), thermoWorldIxs[N_1_wCnt]);

				replica.updWORK() = 0.0;
				replica.upd_WORK_Jacobian() = 0.0;

				// @@@@@@@@@@ LOOP THROUGH NON-EQUILIBRIUM WORLDS --------------------------------------------->
				runReplicaWorldRange(replicaIx, N_1_wCnt, thermoNofWorlds, true);

				// Write log and DCD
				writeReplicaLogAndDCD(mixi, replicaIx, printFreq);

				replica.incrementNofSamples(1);
				thermoState.incrementNofSamples(1);

				SimTK::State& state = worlds.back().integ->updAdvancedState();
				replica.calcZMatrixBAT( worlds.back().getAtomsLocationsInGround( state ));
			
			}

		} // _end_ loop through replicas (NON-EQUILIBRIUM)

		// Mix replicas
		if((getRunType() != RUN_TYPE::DEFAULT) && (nofReplicas != 1)){
			mixReplicas(mixi); // check this
			PrintNofAcceptedSwapsMatrix();
		}else{
			PrintNofAcceptedSwapsMatrix();
		}

		this->nofRounds++; 

	} // end rounds

}




