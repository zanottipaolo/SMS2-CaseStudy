% Owners:
% De Duro Federico      1073477
% Medolago Emanuele     1058907    
% Zanotti Paolo         1074166

T = readtable('Dataset_sanitario.csv');
tNordOvest = T(:, 2:5);
tNordEst = T(:, 6:9);
tCentro = T(:, 10:13);
tSud = T(:, 14:17);
tIsole = T(:, 18:end);

%% Matrice di correlazione
NO_corr = array2table(corr(tNordOvest{:,:}, 'rows','complete'));
NO_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'};
NO_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'}

NE_corr = array2table(corr(tNordEst{:,:}, 'rows','complete'));
NE_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'};
NE_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'}

CE_corr = array2table(corr(tCentro{:,:}, 'rows','complete'));
CE_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'};
CE_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'}

SU_corr = array2table(corr(tSud{:,:}, 'rows','complete'));
SU_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'};
SU_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'}

IS_corr = array2table(corr(tIsole{:,:}, 'rows','complete'));
IS_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'};
IS_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso'}

%% Plot casi di diabete in Italia %%
figure
plot(T.ANNO, T.NO_DIABETE, T.ANNO, T.NE_DIABETE, T.ANNO, T.CE_DIABETE, T.ANNO, T.SU_DIABETE, T.ANNO, T.IS_DIABETE)
title("Casi di Diabete in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
% ------------------------------- %

%% Plot casi di ipertensione in Italia %%
figure
plot(T.ANNO, T.NO_IPERTENSIONE, T.ANNO, T.NE_IPERTENSIONE, T.ANNO, T.CE_IPERTENSIONE, T.ANNO, T.SU_IPERTENSIONE, T.ANNO, T.IS_IPERTENSIONE)
title("Casi di Ipertensione in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Plot casi di Malattie allergiche in Italia %%
figure
plot(T.ANNO, T.NO_MA_ALLERGICHE, T.ANNO, T.NE_MA_ALLERGICHE, T.ANNO, T.CE_MA_ALLERGICHE, T.ANNO, T.SU_MA_ALLERGICHE, T.ANNO, T.IS_MA_ALLERGICHE)
title("Casi di Malattie allergiche in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% OLS per NORDOVEST %%
% Modello Completo
NO_lm1 = fitlm(tNordOvest,'ResponseVar','NO_IPERTENSIONE', 'PredictorVars',{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'});
NO_res = NO_lm1.Residuals.Raw;
%%JB Test residui Nord Ovest
x1=NO_res;
figure
histfit(x1);
title('Residui Nord Ovest');
n=length(x1);
JBdata=(skewness(x1).^2)*n/6+((kurtosis(x1)-3).^2)*n/24;
% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit_NO: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue_NO: ',num2str(pval)]);
disp(['dev std pvalue_NO: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test_NO: ',num2str(potenza)]);

%% OLS per NORDEST
% Modello Completo
NE_lm1 = fitlm(tNordEst,'ResponseVar','NE_IPERTENSIONE', 'PredictorVars',{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'});
NE_res = NE_lm1.Residuals.Raw;
%%JB Test residui Nord Est
x2=NE_res;
figure
histfit(x2);
title('Residui Nord Est');
n=length(x2);
JBdata=(skewness(x2).^2)*n/6+((kurtosis(x2)-3).^2)*n/24;
% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit_NE: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue_NE: ',num2str(pval)]);
disp(['dev std pvalue_NE: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test_NE: ',num2str(potenza)]);

%% OLS per SUD %%
% Modello Completo
SU_lm1 = fitlm(tSud,'ResponseVar','SU_IPERTENSIONE', 'PredictorVars',{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'});
SU_res = SU_lm1.Residuals.Raw;
%%JB Test residui Sud
x3=SU_res;
figure
histfit(x3);
title('Residui Sud');
n=length(x3);
JBdata=(skewness(x3).^2)*n/6+((kurtosis(x3)-3).^2)*n/24;
% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit_SU: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue_SU: ',num2str(pval)]);
disp(['dev std pvalue_SU: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test_SU: ',num2str(potenza)]);

%% OLS per CENTRO %%
% Modello Completo
CE_lm1 = fitlm(tCentro,'ResponseVar','CE_IPERTENSIONE', 'PredictorVars',{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'});
CE_res = CE_lm1.Residuals.Raw;
%%JB Test residui Centro
x4=CE_res;
figure
histfit(x4);
title('Residui Centro');
n=length(x4);
JBdata=(skewness(x4).^2)*n/6+((kurtosis(x4)-3).^2)*n/24;
% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit_CE: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue_CE: ',num2str(pval)]);
disp(['dev std pvalue_CE: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test_CE: ',num2str(potenza)]);

%% OLS per ISOLE %%
% Modello Completo
IS_lm1 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_ECCESSO_PESO','IS_MA_ALLERGICHE'});
% No eccesso di peso
IS_lm2 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_MA_ALLERGICHE'});
IS_res = IS_lm2.Residuals.Raw;
%%JB Test residui Isole
x5=IS_res;
figure
histfit(x5);
title('Residui Isole');
n=length(x5);
JBdata=(skewness(x5).^2)*n/6+((kurtosis(x5)-3).^2)*n/24;
% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit_IS: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue_IS: ',num2str(pval)]);
disp(['dev std pvalue_IS: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test_IS: ',num2str(potenza)]);
