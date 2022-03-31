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
