data {
	int N; //N participants
	matrix[N, 12] rf; //RF items
	matrix[N, 10] intr; //Intrinsic items
	matrix[N, 6] harm; //MFT Harm items
	matrix[N, 6] fair; //MFT Fair items
	matrix[N, 5] group; //MFT group loyalty items
	matrix[N, 6] auth; //MFT respect for authority items
	matrix[N, 6] pure; //MFT pureness items
	vector[N] agent; //In agentic prime condition
	vector[N] inst; //In institutional prime condition
	int rf_enabled;
	int in_enabled;
} 

parameters {
// Latent Factors
matrix[N,5] HFGAP; // Harm, Fairness, Group, Authority, Purity
vector[N] RF; //Religious fundamentalism
vector[N] In; //Intrinsic religiosity

// Lambdas/Loadings
vector<lower=0>[12] lambdaRF;
vector<lower=0>[10] lambdaIntr;
vector<lower=0>[6] lambdaHarm;
vector<lower=0>[6] lambdaFair;
vector<lower=0>[5] lambdaGroup;
vector<lower=0>[6] lambdaAuth;
vector<lower=0>[6] lambdaPure;

// Observed residuals
vector<lower=0>[12] epsilonRF;
vector<lower=0>[10] epsilonIntr;
vector<lower=0>[6] epsilonHarm;
vector<lower=0>[6] epsilonFair;
vector<lower=0>[5] epsilonGroup;
vector<lower=0>[6] epsilonAuth;
vector<lower=0>[6] epsilonPure;

// Nus (add these if necessary; I don't /think/ it's necessary
/*
Beware: The regression is essentially Latent ~ 0 + condition + condition:latent
This means we're setting the latent mean to 0 WHEN in the control condition.
Then Condition is the effect of agent at the average level of latent.
This is fine, just beware of the interpretation.

Changing this to include mean structure for a sec
*/
real nuRF;
vector[5] nuHFGAP;

// Latent correlations or correlated residuals
cholesky_factor_corr[5] phiHFGAP_chol; //Cholesky-decomposed residual correlation matrix for HFGAP

// Structural coefficients
/* Direct effects
A: Agent
I: Institutional
HFGAuP: MFs
 */
real betaAH; 
real betaAF;
real betaAG;
real betaAAu;
real betaAP;

real betaIH; 
real betaIF;
real betaIG;
real betaIAu;
real betaIP;

/* Moderating effects
A: Agent
I: Institutional
In: Intrinsic
*/
real betaAxInH;
real betaAxInF;
real betaAxInG;
real betaAxInAu;
real betaAxInP;

real betaIxInH;
real betaIxInF;
real betaIxInG;
real betaIxInAu;
real betaIxInP;

//real betaAxInRF;
//real betaIxInRF;

/* Conditions to RF, RF to MFs */
real betaAR;
real betaIR;

real betaRH;
real betaRF;
real betaRG;
real betaRAu;
real betaRP;

}

model {
//Convenience params
vector[N] AxIn;
vector[N] IxIn;
//RF_hat
vector[N] RF_hat;
//MF Mus
matrix[N,5] HFGAP_hat;

AxIn = agent .* In; //Agent x Intrinsic
IxIn = inst .* In; //Inst x Intrinsic

RF_hat = nuRF + agent*betaAR + inst*betaIR; //+ AxIn*betaAxInRF + IxIn*betaIxInRF;
HFGAP_hat[,1] = nuHFGAP[1] + agent*betaAH  + inst*betaIH ;
HFGAP_hat[,2] = nuHFGAP[2] + agent*betaAF  + inst*betaIF ;
HFGAP_hat[,3] = nuHFGAP[3] + agent*betaAG  + inst*betaIG ;
HFGAP_hat[,4] = nuHFGAP[4] + agent*betaAAu + inst*betaIAu;
HFGAP_hat[,5] = nuHFGAP[5] + agent*betaAP  + inst*betaIP ;

if(in_enabled){
  HFGAP_hat[,1] = HFGAP_hat[,1]  + AxIn*betaAxInH  + IxIn*betaIxInH;// + RF*betaRH; // H ~ 0 + agent + agent:intrinsic
  HFGAP_hat[,2] = HFGAP_hat[,2]  + AxIn*betaAxInF + IxIn*betaIxInF;// + RF*betaRF; 
  HFGAP_hat[,3] = HFGAP_hat[,3]  + AxIn*betaAxInG + IxIn*betaIxInG;// + RF*betaRG;  //G ~ 0 + inst + inst:intrinsic
  HFGAP_hat[,4] = HFGAP_hat[,4]  + AxIn*betaAxInAu + IxIn*betaIxInAu;// + RF*betaRAu;
  HFGAP_hat[,5] = HFGAP_hat[,5]  + AxIn*betaAxInP +  IxIn*betaIxInP;// + RF*betaRP;
}

if(rf_enabled){
  HFGAP_hat[,1] = HFGAP_hat[,1] + RF*betaRH;
  HFGAP_hat[,2] = HFGAP_hat[,2] + RF*betaRF;
  HFGAP_hat[,3] = HFGAP_hat[,3] + RF*betaRG;
  HFGAP_hat[,4] = HFGAP_hat[,4] + RF*betaRAu;
  HFGAP_hat[,5] = HFGAP_hat[,5] + RF*betaRP;
}

/* Priors 
Note: These priors are under the expectation that the data are STANDARDIZED, except for the dummy-coded conditions
*/
// Latents
In ~ normal(0,1); // Standardized latent
RF ~ normal(RF_hat,1); // Latent with standardized RESIDUAL
for(n in 1:N){
	HFGAP[n] ~ multi_normal_cholesky(HFGAP_hat[n], phiHFGAP_chol); // Correlated Latents with standardized RESIDUALS
}

/* Loadings 
ASSUMING observed variables are STANDARDIZED or centered
*/
lambdaIntr ~ normal(0,1);
lambdaRF ~ normal(0,1);

lambdaHarm ~ normal(0,1);
lambdaFair ~ normal(0,1);
lambdaGroup ~ normal(0,1);
lambdaAuth ~ normal(0,1);
lambdaPure ~ normal(0,1);

// Observed residuals
epsilonIntr ~ normal(0,1);
epsilonRF ~ normal(0,1);

epsilonHarm  ~ normal(0,1);
epsilonFair  ~ normal(0,1);
epsilonGroup ~ normal(0,1);
epsilonAuth  ~ normal(0,1);
epsilonPure  ~ normal(0,1);

// Latent residual covariances
phiHFGAP_chol ~ lkj_corr_cholesky(10); // Omitted because this is a uniform prior, which is default.

// Structural coefficients
betaAR ~ normal(0,1);
betaIR ~ normal(0,1);
betaRH ~ normal(0,1);
betaRF ~ normal(0,1);
betaRG ~ normal(0,1);
betaRAu ~ normal(0,1);
betaRP ~ normal(0,1);

betaAH ~ normal(0,1);
betaAF ~ normal(0,1);
betaAG ~ normal(0,1);
betaAAu ~ normal(0,1);
betaAP ~ normal(0,1);

betaIH ~ normal(0,1);
betaIF ~ normal(0,1);
betaIG ~ normal(0,1);
betaIAu ~ normal(0,1);
betaIP ~ normal(0,1);

betaAxInH ~ normal(0,1);
betaAxInF ~ normal(0,1);
betaAxInG ~ normal(0,1);
betaAxInAu ~ normal(0,1);
betaAxInP ~ normal(0,1);

betaIxInH ~ normal(0,1);
betaIxInF ~ normal(0,1);
betaIxInG ~ normal(0,1);
betaIxInAu ~ normal(0,1);
betaIxInP ~ normal(0,1);

//betaAxInRF ~ normal(0,1);
//betaIxInRF ~ normal(0,1);

//Nus 
nuRF ~ normal(0,1);
nuHFGAP ~ normal(0,1);


/* Likelihoods */
// Intrinsic
for(j in 1:10) {
	intr'[j] ~ normal(lambdaIntr[j]*In', epsilonIntr[j]);
}
// RF
for(j in 1:12){
	rf'[j] ~ normal(lambdaRF[j]*RF', epsilonRF[j]);
}
// MFTs
for(j in 1:5){
	//Group
	group'[j] ~ normal(lambdaGroup[j]*HFGAP'[3],epsilonGroup[j]);
}
for(j in 1:6){
	//harm,fair,auth,pure
	harm'[j] ~ normal(lambdaHarm[j]*HFGAP'[1],epsilonHarm[j]);
	fair'[j] ~ normal(lambdaFair[j]*HFGAP'[2],epsilonFair[j]);
	auth'[j] ~ normal(lambdaAuth[j]*HFGAP'[4],epsilonAuth[j]);
	pure'[j] ~ normal(lambdaPure[j]*HFGAP'[5],epsilonPure[j]);
}

}

generated quantities {
/* Declarations */
// Standardized structural coefficients
real betaRH_std;
real betaRF_std;
real betaRG_std;
real betaRAu_std;
real betaRP_std;

// Effect -> dummy code
real betaARDummy = (nuRF + betaAR) - (nuRF - betaAR - betaIR); // mu_agent - mu_control
real betaIRDummy = (nuRF + betaIR) - (nuRF - betaAR - betaIR); // mu_inst - mu_control
vector[5] betaAHFGAPDummy;
vector[5] betaIHFGAPDummy;

// Effect -> +1SD dummy code
vector[5] betaAHFGAPDummyInxn = [0, 0, 0, 0, 0]';
vector[5] betaIHFGAPDummyInxn = [0, 0, 0, 0, 0]';

// Indirect effects
real ARH;
real ARF;
real ARG;
real ARA;
real ARP;
real IRH;
real IRF;
real IRG;
real IRA;
real IRP;
 
 //Residual Correlation
corr_matrix[5] phiHFGAP_cor;
 
betaAHFGAPDummy[1] = (nuHFGAP[1] + betaAH) - (nuHFGAP[1] - betaAH - betaIH);
betaAHFGAPDummy[2] = (nuHFGAP[2] + betaAF) - (nuHFGAP[2] - betaAF - betaIF);
betaAHFGAPDummy[3] = (nuHFGAP[3] + betaAG) - (nuHFGAP[3] - betaAG - betaIG);
betaAHFGAPDummy[4] = (nuHFGAP[4] + betaAAu) - (nuHFGAP[4] - betaAAu - betaIAu);
betaAHFGAPDummy[5] = (nuHFGAP[5] + betaAP) - (nuHFGAP[5] - betaAP - betaIP);

betaIHFGAPDummy[1] = (nuHFGAP[1] + betaIH) - (nuHFGAP[1] - betaAH - betaIH);
betaIHFGAPDummy[2] = (nuHFGAP[2] + betaIF) - (nuHFGAP[2] - betaAF - betaIF);
betaIHFGAPDummy[3] = (nuHFGAP[3] + betaIG) - (nuHFGAP[3] - betaAG - betaIG);
betaIHFGAPDummy[4] = (nuHFGAP[4] + betaIAu) - (nuHFGAP[4] - betaAAu - betaIAu);
betaIHFGAPDummy[5] = (nuHFGAP[5] + betaIP) - (nuHFGAP[5] - betaAP - betaIP);


if(in_enabled){
  betaAHFGAPDummyInxn[1] = (nuHFGAP[1] + betaAH + betaAxInH) - (nuHFGAP[1] - betaAH - betaIH - betaAxInH - betaIxInH) - betaAHFGAPDummy[1];
  betaAHFGAPDummyInxn[2] = (nuHFGAP[2] + betaAF + betaAxInF) - (nuHFGAP[2] - betaAF - betaIF - betaAxInF - betaIxInF) - betaAHFGAPDummy[2];
  betaAHFGAPDummyInxn[3] = (nuHFGAP[3] + betaAG + betaAxInG) - (nuHFGAP[3] - betaAG - betaIG - betaAxInG - betaIxInG) - betaAHFGAPDummy[3];
  betaAHFGAPDummyInxn[4] = (nuHFGAP[4] + betaAAu + betaAxInAu) - (nuHFGAP[4] - betaAAu - betaIAu - betaAxInAu - betaIxInAu) - betaAHFGAPDummy[4];
  betaAHFGAPDummyInxn[5] = (nuHFGAP[5] + betaAP + betaAxInP) - (nuHFGAP[5] - betaAP - betaIP - betaAxInP - betaIxInP) - betaAHFGAPDummy[5];
  
  betaIHFGAPDummyInxn[1] = (nuHFGAP[1] + betaIH + betaIxInH) - (nuHFGAP[1] - betaAH - betaIH - betaAxInH - betaIxInH) - betaIHFGAPDummy[1];
  betaIHFGAPDummyInxn[2] = (nuHFGAP[2] + betaIF + betaIxInF) - (nuHFGAP[2] - betaAF - betaIF - betaAxInF - betaIxInF) - betaIHFGAPDummy[1];
  betaIHFGAPDummyInxn[3] = (nuHFGAP[3] + betaIG + betaIxInG) - (nuHFGAP[3] - betaAG - betaIG - betaAxInG - betaIxInG) - betaIHFGAPDummy[1];
  betaIHFGAPDummyInxn[4] = (nuHFGAP[4] + betaIAu + betaIxInAu) - (nuHFGAP[4] - betaAAu - betaIAu - betaAxInAu - betaIxInAu) - betaIHFGAPDummy[1];
  betaIHFGAPDummyInxn[5] = (nuHFGAP[5] + betaIP + betaIxInP) - (nuHFGAP[5] - betaAP - betaIP - betaAxInH - betaIxInP) - betaIHFGAPDummy[1];
}

// Compute using temporary, non-saved variables
{
  vector[N] AxIn = agent .* In;
  vector[N] IxIn = inst .* In;
  vector[N] RF_hat;
  matrix[N,5] HFGAP_hat;
  real variance_H ;
  real variance_F ;
  real variance_G ;
  real variance_Au;
  real variance_P ;
  real variance_RF;
  
  RF_hat = nuRF + agent*betaAR + inst*betaIR; //+ AxIn*betaAxInRF + IxIn*betaIxInRF;
  HFGAP_hat[,1] = nuHFGAP[1] + agent*betaAH  + inst*betaIH ;
  HFGAP_hat[,2] = nuHFGAP[2] + agent*betaAF  + inst*betaIF ;
  HFGAP_hat[,3] = nuHFGAP[3] + agent*betaAG  + inst*betaIG ;
  HFGAP_hat[,4] = nuHFGAP[4] + agent*betaAAu + inst*betaIAu;
  HFGAP_hat[,5] = nuHFGAP[5] + agent*betaAP  + inst*betaIP ;
  
  if(in_enabled){
    HFGAP_hat[,1] = HFGAP_hat[,1]  + AxIn*betaAxInH  + IxIn*betaIxInH;// + RF*betaRH; // H ~ 0 + agent + agent:intrinsic
    HFGAP_hat[,2] = HFGAP_hat[,2]  + AxIn*betaAxInF + IxIn*betaIxInF;// + RF*betaRF; 
    HFGAP_hat[,3] = HFGAP_hat[,3]  + AxIn*betaAxInG + IxIn*betaIxInG;// + RF*betaRG;  //G ~ 0 + inst + inst:intrinsic
    HFGAP_hat[,4] = HFGAP_hat[,4]  + AxIn*betaAxInAu + IxIn*betaIxInAu;// + RF*betaRAu;
    HFGAP_hat[,5] = HFGAP_hat[,5]  + AxIn*betaAxInP +  IxIn*betaIxInP;// + RF*betaRP;
  }
  
  if(rf_enabled){
    HFGAP_hat[,1] = HFGAP_hat[,1] + RF*betaRH;
    HFGAP_hat[,2] = HFGAP_hat[,2] + RF*betaRF;
    HFGAP_hat[,3] = HFGAP_hat[,3] + RF*betaRG;
    HFGAP_hat[,4] = HFGAP_hat[,4] + RF*betaRAu;
    HFGAP_hat[,5] = HFGAP_hat[,5] + RF*betaRP;
  }
  
  variance_H = variance(HFGAP_hat[,1]) + 1; //Var(H) = Var(predicted) + Var(residual) [= 1]
  variance_F = variance(HFGAP_hat[,2]) + 1;
  variance_G = variance(HFGAP_hat[,3]) + 1;
  variance_Au = variance(HFGAP_hat[,4]) + 1;
  variance_P = variance(HFGAP_hat[,5]) + 1;
  variance_RF = variance(RF_hat) + 1;
  
  betaRH_std = betaRH*sqrt(variance_RF)/sqrt(variance_H); //Beta = beta*sd(x)/sd(y)
  betaRF_std = betaRF*sqrt(variance_RF)/sqrt(variance_F);
  betaRG_std = betaRG*sqrt(variance_RF)/sqrt(variance_G);
  betaRAu_std = betaRAu*sqrt(variance_RF)/sqrt(variance_Au);
  betaRP_std = betaRP*sqrt(variance_RF)/sqrt(variance_P);
}

// Marginal Indirect effects; on standardized scale
ARH = betaARDummy*betaRH_std;
ARF = betaARDummy*betaRF_std;
ARG = betaARDummy*betaRG_std;
ARA = betaARDummy*betaRAu_std;
ARP = betaARDummy*betaRP_std;
IRH = betaIRDummy*betaRH_std;
IRF = betaIRDummy*betaRF_std;
IRG = betaIRDummy*betaRG_std;
IRA = betaIRDummy*betaRAu_std;
IRP = betaIRDummy*betaRP_std;

// Residual correlation matrix
phiHFGAP_cor = multiply_lower_tri_self_transpose(phiHFGAP_chol);

}
