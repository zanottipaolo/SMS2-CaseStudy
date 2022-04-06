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

%% Plot casi di Eccesso di peso in Italia %%
figure
plot(T.ANNO, T.NO_ECCESSO_PESO, T.ANNO, T.NE_ECCESSO_PESO, T.ANNO, T.CE_ECCESSO_PESO, T.ANNO, T.SU_ECCESSO_PESO, T.ANNO, T.IS_ECCESSO_PESO)
title("Casi di Eccesso di peso in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%


%% OLS per NORDOVEST %%

% Modello Completo
NO_lm1 = fitlm(tNordOvest,'ResponseVar','NO_IPERTENSIONE', 'PredictorVars',{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'});
NO_res = NO_lm1.Residuals.Raw;

plot(NO_lm1);

%% STIMA DELLA Y
Y = tNordOvest.NO_IPERTENSIONE; % Vera Y
Y = fillmissing(Y,'linear') % Riempio i NaN con valori lineari
n = length(Y);
X = [ones(n, 1) fillmissing(tNordOvest.NO_DIABETE,'linear') fillmissing(tNordOvest.NO_ECCESSO_PESO,'linear') fillmissing(tNordOvest.NO_MA_ALLERGICHE,'linear')];
X = X(sum(isnan(X),2)==0,:);
X = fillmissing(X, 'linear');

% Verifica che il det(X'X) > 0
det(X'*X);

length(Y)
length(X)

% Stima di beta hat e y hat
NO_B_hat = (X'*X)\X'*Y; %1°: Intercetta
NO_y_hat = X*B_hat;

% Plotto la reale y con quella stimata
plot(T.ANNO, y_hat, T.ANNO, tNordOvest.NO_IPERTENSIONE)
title("Ipertensione stimata - Ipertensione reale")
legend("Y HAT", "Y REAL")
xlabel("Anno")
ylabel("Casi di ipertensione (%)")
grid;

% Calcolo di devianza totale, residua, spiegata e di R^2 manualmente
mY = mean(Y);
Dtot = sum((Y-mY).^2);
Dres = sum((Y-y_hat).^2);
Dsp = sum((y_hat-mY).^2);
R2 = 1-(Dres/Dtot);

% Calcolo dello scarto quadratico medio
k = 2;
s2e = Dres/(n - k - 1);
s = sqrt(s2e);

% Plot residui dal fitlm e dai Min Quad. manualmente calcolati
residuals = Y - y_hat;
plot([1:length(residuals)], residuals, [1:length(NO_res)], NO_res)
title("Residui OLS fitlm - OLS manuali")
legend("Manuali", "fitlm")
xlabel("Osservazione")
ylabel("Residuo")
grid;

alpha = .05;
[h1,p1,jbstat1,critval1] = jbtest(residuals, alpha);

%Matrice δ
delta = (X'*residuals)/n;

% Delta ha tutti i componenti ≃ 0, quindi si può concludere che la stima
% dei Beta sia non distorta.

% JB Test residui Nord Ovest
x1 = NO_res;
figure
histfit(x1);
title('Residui Nord Ovest');
n = length(x1);
JBdata = (skewness(x1).^2)*n/6+((kurtosis(x1)-3).^2)*n/24;

% Simulazione MC
m = 1000;
X0 = randn(m,n);
JB0 = (skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha = 0.05;
JBcrit = prctile(JB0,100*(1-alpha));
disp(['JBcrit_NO: ',num2str(JBcrit)]);
pval = mean(JB0>JBdata);
stdp = sqrt(pval*(1-pval)/m);
disp(['pvalue_NO: ',num2str(pval)]);
disp(['dev std pvalue_NO: ',num2str(stdp)]);
X1 = chi2rnd(2,m,n);
JB1 = (skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza = mean(JB1>JBcrit);
disp(['potenza test_NO: ',num2str(potenza)]);

% 1. Grafico dei residui (media uguale a 0)
plot(NO_res)
ylabel('Residui')
xlabel('Osservazioni')
yline(nanmean(NO_res), 'Color', 'b', 'LineWidth', 3)
title('Grafico dei residui - Nord Ovest')

% 2. Andamento dei Percentili
qqplot(NO_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati')

% 3. Incorrelazione dei regressori con i residui
[S,AX,BigAx,H,HAx] = plotmatrix(tNordOvest{:,{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'}}, NO_res)
title 'Correlazione Residui-Regressori'
AX(1,1).YLabel.String = 'Residui'
AX(1,1).XLabel.String = 'DIABETE'
AX(1,2).XLabel.String = 'ECCESSO DI PESO'
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE'

% Verifica dell'incorrelazione tramite gli indici di correlazione
NO_mat_corr_residui = corrcoef([NO_res, tNordOvest.NO_DIABETE,...
    tNordOvest.NO_ECCESSO_PESO, tNordOvest.NO_MA_ALLERGICHE], 'Rows','complete');
NO_res_corr_w_reg = NO_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

autocorr(NO_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione")

% 4. Ricerca degli outliers
residui_studentizzati = NO_lm1.Residuals.Studentized;
scatter(NO_lm1.Fitted, residui_studentizzati)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')

% 5. Varianza dei residui
plotResiduals(NO_lm1, 'fitted', 'Marker','o')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(NO_lm1,'exact','both')

% ECM Test
[Mean,Covariance] = ecmnmle(tNordOvest.NO_DIABETE)

%% OLS per NORDEST

% Modello Completo
NE_lm1 = fitlm(tNordEst,'ResponseVar','NE_IPERTENSIONE', 'PredictorVars',{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'});
NE_res = NE_lm1.Residuals.Raw;

plot(NE_lm1);

% JB Test residui Nord Est
x2 = NE_res;
figure
histfit(x2);
title('Residui Nord Est');
n = length(x2);
JBdata = (skewness(x2).^2)*n/6+((kurtosis(x2)-3).^2)*n/24;

% Simulazione MC
m = 1000;
X0 = randn(m,n);
JB0 = (skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha = 0.05;
JBcrit = prctile(JB0,100*(1-alpha));
disp(['JBcrit_NE: ',num2str(JBcrit)]);
pval = mean(JB0>JBdata);
stdp = sqrt(pval*(1-pval)/m);
disp(['pvalue_NE: ',num2str(pval)]);
disp(['dev std pvalue_NE: ',num2str(stdp)]);
X1 = chi2rnd(2,m,n);
JB1 = (skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza = mean(JB1>JBcrit);
disp(['potenza test_NE: ',num2str(potenza)]);

% 1. Grafico dei residui (media uguale a 0)
plot(NE_res)
ylabel('Residui')
xlabel('Osservazioni')
yline(nanmean(NE_res), 'Color', 'b', 'LineWidth', 3)
title('Grafico dei residui - Nord Est')

% 2. Andamento dei Percentili
qqplot(NE_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati')

% 3. Incorrelazione dei regressori con i residui
[S,AX,BigAx,H,HAx] = plotmatrix(tNordEst{:,{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'}}, NE_res)
title 'Correlazione Residui-Regressori'
AX(1,1).YLabel.String = 'Residui'
AX(1,1).XLabel.String = 'DIABETE'
AX(1,2).XLabel.String = 'ECCESSO DI PESO'
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE'

% Verifica dell'incorrelazione tramite gli indici di correlazione
NE_mat_corr_residui = corrcoef([NE_res, tNordEst.NE_DIABETE,...
    tNordEst.NE_ECCESSO_PESO, tNordEst.NE_MA_ALLERGICHE], 'Rows','complete');
NE_res_corr_w_reg = NE_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

autocorr(NE_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione")

% 4. Ricerca degli outliers
NE_residui_stud = NE_lm1.Residuals.Studentized;
scatter(NE_lm1.Fitted, NE_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')

% 5. Varianza dei residui
plotResiduals(NE_lm1, 'fitted', 'Marker','o')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(NE_lm1,'exact','both')

%% OLS per CENTRO %%

% Modello Completo
CE_lm1 = fitlm(tCentro,'ResponseVar','CE_IPERTENSIONE', 'PredictorVars',{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'});
CE_res = CE_lm1.Residuals.Raw;

plot(CE_lm1);

% JB Test residui Centro
x4 = CE_res;
figure
histfit(x4);
title('Residui Centro');
n = length(x4);
JBdata = (skewness(x4).^2)*n/6+((kurtosis(x4)-3).^2)*n/24;

% Simulazione MC
m = 1000;
X0 = randn(m,n);
JB0 = (skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha = 0.05;
JBcrit = prctile(JB0,100*(1-alpha));
disp(['JBcrit_CE: ',num2str(JBcrit)]);
pval = mean(JB0>JBdata);
stdp = sqrt(pval*(1-pval)/m);
disp(['pvalue_CE: ',num2str(pval)]);
disp(['dev std pvalue_CE: ',num2str(stdp)]);
X1 = chi2rnd(2,m,n);
JB1 = (skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza = mean(JB1>JBcrit);
disp(['potenza test_CE: ',num2str(potenza)]);

% 1. Grafico dei residui (media uguale a 0)
plot(CE_res)
ylabel('Residui')
xlabel('Osservazioni')
yline(nanmean(CE_res), 'Color', 'b', 'LineWidth', 3)
title('Grafico dei residui - Centro')

% 2. Andamento dei Percentili
qqplot(CE_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati')

% 3. Incorrelazione dei regressori con i residui
[S,AX,BigAx,H,HAx] = plotmatrix(tCentro{:,{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'}}, CE_res)
title 'Correlazione Residui-Regressori'
AX(1,1).YLabel.String = 'Residui'
AX(1,1).XLabel.String = 'DIABETE'
AX(1,2).XLabel.String = 'ECCESSO DI PESO'
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE'

% Verifica dell'incorrelazione tramite gli indici di correlazione
CE_mat_corr_residui = corrcoef([CE_res, tCentro.CE_DIABETE,...
    tCentro.CE_ECCESSO_PESO, tCentro.CE_MA_ALLERGICHE], 'Rows','complete');
CE_res_corr_w_reg = CE_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

autocorr(CE_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione")

% 4. Ricerca degli outliers
CE_residui_stud = CE_lm1.Residuals.Studentized;
scatter(CE_lm1.Fitted, CE_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')

% 5. Varianza dei residui
plotResiduals(CE_lm1, 'fitted', 'Marker','o')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(CE_lm1,'exact','both')

%% OLS per SUD %%
% Modello Completo
SU_lm1 = fitlm(tSud,'ResponseVar','SU_IPERTENSIONE', 'PredictorVars',{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'});
SU_res = SU_lm1.Residuals.Raw;

plot(SU_lm1);

% JB Test residui Sud
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

% 1. Grafico dei residui (media uguale a 0)
plot(SU_res)
ylabel('Residui')
xlabel('Osservazioni')
yline(nanmean(SU_res), 'Color', 'b', 'LineWidth', 3)
title('Grafico dei residui - Sud')

% 2. Andamento dei Percentili
qqplot(SU_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati')

% 3. Incorrelazione dei regressori con i residui
[S,AX,BigAx,H,HAx] = plotmatrix(tSud{:,{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'}}, SU_res)
title 'Correlazione Residui - Regressori'
AX(1,1).YLabel.String = 'Residui'
AX(1,1).XLabel.String = 'DIABETE'
AX(1,2).XLabel.String = 'ECCESSO DI PESO'
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE'

% Verifica dell'incorrelazione tramite gli indici di correlazione
SU_mat_corr_residui = corrcoef([SU_res, tSud.SU_DIABETE,...
    tSud.SU_ECCESSO_PESO, tSud.SU_MA_ALLERGICHE], 'Rows','complete');
SU_res_corr_w_reg = SU_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

autocorr(SU_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione")

% 4. Ricerca degli outliers
SU_residui_stud = SU_lm1.Residuals.Studentized;
scatter(SU_lm1.Fitted, SU_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')

% 5. Varianza dei residui
plotResiduals(SU_lm1, 'fitted', 'Marker','o')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(SU_lm1,'exact','both')

%% OLS per ISOLE %%

% Modello Completo
IS_lm1 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_ECCESSO_PESO','IS_MA_ALLERGICHE'});

% No eccesso di peso (modello utilizzato)
IS_lm2 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_MA_ALLERGICHE'});
IS_res = IS_lm2.Residuals.Raw;

plot(IS_lm2);

% Massima verosomiglianza
% histogram(IS_res)
% mle(tIsole.IS_DIABETE, 'distribution','Normal', 'Alpha', .01)

% JB Test residui Isole
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

% 1. Grafico dei residui (media uguale a 0)
plot(IS_res)
ylabel('Residui')
xlabel('Osservazioni')
yline(nanmean(IS_res), 'Color', 'b', 'LineWidth', 3)
title('Grafico dei residui - Isole')

% 2. Andamento dei Percentili
qqplot(IS_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati')

% 3. Incorrelazione dei regressori con i residui
[S,AX,BigAx,H,HAx] = plotmatrix(tIsole{:,{'IS_DIABETE','IS_ECCESSO_PESO','IS_MA_ALLERGICHE'}}, IS_res)
title 'Correlazione Residui - Regressori'
AX(1,1).YLabel.String = 'Residui'
AX(1,1).XLabel.String = 'DIABETE'
AX(1,2).XLabel.String = 'ECCESSO DI PESO'
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE'

% Verifica dell'incorrelazione tramite gli indici di correlazione
IS_mat_corr_residui = corrcoef([IS_res, tIsole.IS_DIABETE,...
    tIsole.IS_ECCESSO_PESO, tIsole.IS_MA_ALLERGICHE], 'Rows','complete');
IS_res_corr_w_reg = IS_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

autocorr(IS_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione")

% 4. Ricerca degli outliers
IS_residui_stud = IS_lm2.Residuals.Studentized;
scatter(IS_lm2.Fitted, IS_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')

% 5. Varianza dei residui
plotResiduals(IS_lm2, 'fitted', 'Marker','o')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(IS_lm1,'exact','both')

close all