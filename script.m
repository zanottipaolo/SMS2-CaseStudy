% Owners:
% De Duro Federico      1073477
% Medolago Emanuele     1058907    
% Zanotti Paolo         1074166

T = readtable('Dataset_sanitario.csv')
tNordOvest = T(:, 2:7);
tNordEst = T(:, 8:13);
tCentro = T(:, 14:19);
tSud = T(:, 20:25);
tIsole = T(:, 26:31);

%% Matrice di correlazione
NO_corr = array2table(corr(tNordOvest{:,:}, 'rows','complete'));
NO_corr.Properties.VariableNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'};
NO_corr.Properties.RowNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'}

NE_corr = array2table(corr(tNordEst{:,:}, 'rows','complete'));
NE_corr.Properties.VariableNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'};
NE_corr.Properties.RowNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'}

CE_corr = array2table(corr(tCentro{:,:}, 'rows','complete'));
CE_corr.Properties.VariableNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'};
CE_corr.Properties.RowNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'}

SU_corr = array2table(corr(tSud{:,:}, 'rows','complete'));
SU_corr.Properties.VariableNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'};
SU_corr.Properties.RowNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'}

IS_corr = array2table(corr(tIsole{:,:}, 'rows','complete'));
IS_corr.Properties.VariableNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'};
IS_corr.Properties.RowNames = {'Diabete', 'Ipertensione', 'Tumori', 'Fumatori', 'Peso', 'Alcool'}

heatmap(NO_corr)

% Completo
lm1 = fitlm(tSud,'ResponseVar','SU_M_TUMORI', 'PredictorVars',{'SU_DIABETE',...
    'SU_IPERTENSIONE','SU_FUMATORI', 'SU_ECCESSO_PESO','SU_ALCOOL'});

% No fumatori
lm2 = fitlm(tSud,'ResponseVar','SU_M_TUMORI', 'PredictorVars',{'SU_DIABETE',...
    'SU_IPERTENSIONE', 'SU_ECCESSO_PESO','SU_ALCOOL'});

% No diabete
lm3 = fitlm(tSud,'ResponseVar','SU_M_TUMORI', 'PredictorVars',{'SU_IPERTENSIONE', ...
    'SU_ECCESSO_PESO','SU_ALCOOL'});

% No eccesso di peso
lm4 = fitlm(tSud,'ResponseVar','SU_M_TUMORI', 'PredictorVars',{'SU_IPERTENSIONE', ...
    'SU_ALCOOL'});

stepwise_linear = stepwiselm(tSud,'Upper','linear', 'ResponseVar','SU_M_TUMORI','PEnter', 0.05)

res = lm4.Residuals.Raw
histfit(res)

%% Plot casi di diabete in Italia %%
figure
plot(T.ANNO, T.NO_DIABETE, T.ANNO, T.NE_DIABETE, T.ANNO, T.CE_DIABETE, T.ANNO, T.SU_DIABETE, T.ANNO, T.IS_DIABETE)
title("Casi di Diabete in Italia 1990 - 2013")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
% ------------------------------- %

%% Plot casi di ipertensione in Italia %%
figure
plot(T.ANNO, T.NO_IPERTENSIONE, T.ANNO, T.NE_IPERTENSIONE, T.ANNO, T.CE_IPERTENSIONE, T.ANNO, T.SU_IPERTENSIONE, T.ANNO, T.IS_IPERTENSIONE)
title("Casi di Ipertensione in Italia 1990 - 2013")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Plot casi di tumori in Italia %%
figure
plot(T.ANNO, T.NO_M_TUMORI, T.ANNO, T.NE_M_TUMORI, T.ANNO, T.CE_M_TUMORI, T.ANNO, T.SU_M_TUMORI, T.ANNO, T.IS_M_TUMORI)
title("Casi di Tumori in Italia 1990 - 2013")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Plot fumatori in Italia %%
figure
plot(T.ANNO, T.NO_FUMATORI, T.ANNO, T.NE_FUMATORI, T.ANNO, T.CE_FUMATORI, T.ANNO, T.SU_FUMATORI, T.ANNO, T.IS_FUMATORI)
title("Fumatori in Italia 1990 - 2013")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Matrice di grafici di correlazione
[S,AX,BigAx,H,HAx] = plotmatrix(tNordOvest);
title 'Matrice Grafici Per Analisi Correlazione NORD OVEST';
AX(1,1).YLabel.String = 'Diabete';
AX(2,1).YLabel.String = 'Ipertensione';
AX(3,1).YLabel.String = 'Tumori';
AX(4,1).YLabel.String = 'Fumatori';
AX(5,1).YLabel.String = 'Eccesso peso';
AX(6,1).YLabel.String = 'Alcool';

AX(6,1).XLabel.String = 'Diabete'; 
AX(6,2).XLabel.String = 'Ipertensione'; 
AX(6,3).XLabel.String = 'Tumori';
AX(6,4).XLabel.String = 'Fumatori';
AX(6,5).XLabel.String = 'Eccesso peso';
AX(6,6).XLabel.String = 'Alcool';

%% JB Test
%x=Datasetsanitariocompleto.SU_M_TUMORI;
x = res
histfit(x);
n=length(x);
JBdata=(skewness(x).^2)*n/6+((kurtosis(x)-3).^2)*n/24;
%% Simulazione MC
m=1000;
X0=randn(m,n);
JB0=(skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha=0.05;
JBcrit=prctile(JB0,100*(1-alpha));
disp(['JBcrit: ',num2str(JBcrit)]);
pval=mean(JB0>JBdata);
stdp=sqrt(pval*(1-pval)/m);
disp(['pvalue: ',num2str(pval)]);
disp(['dev std pvalue: ',num2str(stdp)]);
X1=chi2rnd(2,m,n);
JB1=(skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza=mean(JB1>JBcrit);
disp(['potenza test: ',num2str(potenza)]);
