% Owners:
% De Duro Federico      1073477
% Medolago Emanuele     1058907    
% Zanotti Paolo         1074166

% Dependencies
% 1. Mskekur (Mardia test)
% 2. TestHet (heteroskedasticity)
% 3. Statistics and Machine Learning Toolbox
% 4. Econometrics Toolbox (for 'autocorr')

addpath("Mskekur\")
addpath("tHet.m\")

close all
clearvars
clc

load('dataset.mat');
tNordOvest = T(:, 2:7);
tNordEst = T(:, 8:13);
tCentro = T(:, 14:19);
tSud = T(:, 20:25);
tIsole = T(:, 26:end);

%% Stima dei dati mancanti tramite Mardia - Non soddisfacente
Mardia_tNordOvest = tNordOvest;
rows = any(isnan(tNordOvest{:,:}),2);
Mardia_tNordOvest(rows,:) = [];
Mardia_tNordOvest(:,3) = [];

x1 = Mardia_tNordOvest.NO_DIABETE;
x2 = Mardia_tNordOvest.NO_MA_ALLERGICHE;
x3 = Mardia_tNordOvest.NO_SEDENTARI;
x4 = Mardia_tNordOvest.NO_ECCESSO_PESO;
x5 = Mardia_tNordOvest.NO_MA_RESPIRATORIE;
x = [x1 x2 x3 x4 x5]

Mskekur(x, 1, 0.05)

%% Stima dei dati mancanti con media mobile
T_Stimata = T;
steps = 7;
for i = 1:width(T_Stimata)
    for j = 1:height(T_Stimata)
        if isnan(T_Stimata{j,i})
            lower = j - steps;
            upper = j + steps;

            if lower < 1
                lower = 1;
            end
            if upper > width(T_Stimata)
                upper = width(T_Stimata);
            end
            somma = 0;
            count = 0;
            for k = lower:upper
                if isnan(T_Stimata{k, i})  
                else
                    somma = somma + T_Stimata{k, i};
                    count = count + 1;
                end
            end
            T_Stimata{j,i} = somma / count;
        end
    end
end

% Plot del dataset dato, con quello stimato
subplot(2,1,1)
x = T.ANNO;
y1 = T_Stimata{:,2:end};
plot(x,y1)

subplot(2,1,2); 
y2 = T{:,2:end};
plot(x,y2)

close all

tNordOvest = T_Stimata(:, 2:7);
tNordEst = T_Stimata(:, 8:13);
tCentro = T_Stimata(:, 14:19);
tSud = T_Stimata(:, 20:25);
tIsole = T_Stimata(:, 26:end);

%% Matrici di correlazione
NO_corr = array2table(corr(tNordOvest{:,:}, 'rows','complete'));
NO_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'};
NO_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'}

NE_corr = array2table(corr(tNordEst{:,:}, 'rows','complete'));
NE_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'};
NE_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'}

PM = tNordEst;
[S,AX,BigAx,H,HAx] = plotmatrix(PM{:,:});
title 'Matrice Grafici di correlazione';
AX(1,1).YLabel.String = 'Diabete'; 
AX(2,1).YLabel.String = 'Ma. allergiche';
AX(3,1).YLabel.String = 'Ipertensione'; 
AX(4,1).YLabel.String = 'Ec. di peso'; 
AX(5,1).YLabel.String = 'Ma. respiratorie'; 
AX(6,1).YLabel.String = 'Sedentari';

AX(6,1).XLabel.String = 'Diabete'; 
AX(6,2).XLabel.String = 'Ma. allergiche';
AX(6,3).XLabel.String = 'Ipertensione'; 
AX(6,4).XLabel.String = 'Ec. di peso'; 
AX(6,5).XLabel.String = 'Ma. respiratorie'; 
AX(6,6).XLabel.String = 'Sedentari';

S(3,1).Color = 'r';
S(3,2).Color = 'r';
S(3,4).Color = 'r';
S(3,5).Color = 'r';
S(3,6).Color = 'r';

CE_corr = array2table(corr(tCentro{:,:}, 'rows','complete'));
CE_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'};
CE_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'}

SU_corr = array2table(corr(tSud{:,:}, 'rows','complete'));
SU_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'};
SU_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'}

IS_corr = array2table(corr(tIsole{:,:}, 'rows','complete'));
IS_corr.Properties.VariableNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'};
IS_corr.Properties.RowNames = {'Diabete','Malattie Allergiche', 'Ipertensione','Peso','Malattie Respiratorie', 'Sedentari'}

%% Plot casi di diabete in Italia %%
figure
plot(T_Stimata.ANNO, T_Stimata.NO_DIABETE, T_Stimata.ANNO, T_Stimata.NE_DIABETE, T_Stimata.ANNO, T_Stimata.CE_DIABETE, T_Stimata.ANNO, T_Stimata.SU_DIABETE, T_Stimata.ANNO, T_Stimata.IS_DIABETE)
title("Casi di Diabete in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
% ------------------------------- %

%% Plot casi di ipertensione in Italia %%
figure
plot(T_Stimata.ANNO, T_Stimata.NO_IPERTENSIONE, T_Stimata.ANNO, T_Stimata.NE_IPERTENSIONE, T_Stimata.ANNO, T_Stimata.CE_IPERTENSIONE, T_Stimata.ANNO, T_Stimata.SU_IPERTENSIONE, T_Stimata.ANNO, T_Stimata.IS_IPERTENSIONE)
title("Casi di Ipertensione in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Plot casi di Malattie allergiche in Italia %%
figure
plot(T_Stimata.ANNO, T_Stimata.NO_MA_ALLERGICHE, T_Stimata.ANNO, T_Stimata.NE_MA_ALLERGICHE, T_Stimata.ANNO, T_Stimata.CE_MA_ALLERGICHE, T_Stimata.ANNO, T_Stimata.SU_MA_ALLERGICHE, T_Stimata.ANNO, T_Stimata.IS_MA_ALLERGICHE)
title("Casi di Malattie allergiche in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

%% Plot casi di Eccesso di peso in Italia %%
figure
plot(T.ANNO, T_Stimata.NO_ECCESSO_PESO, T_Stimata.ANNO, T_Stimata.NE_ECCESSO_PESO, T_Stimata.ANNO, T_Stimata.CE_ECCESSO_PESO, T_Stimata.ANNO, T_Stimata.SU_ECCESSO_PESO, T_Stimata.ANNO, T_Stimata.IS_ECCESSO_PESO)
title("Casi di Eccesso di peso in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")

%% Plot incidenza Sedentari in Italia %%
figure
plot(T_Stimata.ANNO, T_Stimata.NO_SEDENTARI, T_Stimata.ANNO, T_Stimata.NE_SEDENTARI, T_Stimata.ANNO, T_Stimata.CE_SEDENTARI, T_Stimata.ANNO, T_Stimata.SU_SEDENTARI, T_Stimata.ANNO, T_Stimata.IS_SEDENTARI)
title("Sedentarietà in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")

%% Plot casi di Malattie Respiratorie croniche in Italia %%
figure
plot(T_Stimata.ANNO, T_Stimata.NO_MA_RESPIRATORIE, T_Stimata.ANNO, T_Stimata.NE_MA_RESPIRATORIE, T_Stimata.ANNO, T_Stimata.CE_MA_RESPIRATORIE, T_Stimata.ANNO, T_Stimata.SU_MA_RESPIRATORIE, T_Stimata.ANNO, T_Stimata.IS_MA_RESPIRATORIE)
title("Casi di Malattie respiratorie croniche in Italia 1990 - 2014")
legend("Nord Ovest", "Nord Est", "Centro", "Sud", "Isole")
%% ------------------------------- %%

close all

%% OLS per NORDOVEST %%
%%CROSS VALIDAZIONE Nord Ovest
% creazione regressori
v=ones(25,1);
x1NO=[v tNordOvest.NO_ECCESSO_PESO];
x2NO=[v tNordOvest.NO_ECCESSO_PESO tNordOvest.NO_DIABETE];
x3NO=[v tNordOvest.NO_ECCESSO_PESO tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE];
x4NO=[v tNordOvest.NO_ECCESSO_PESO tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE tNordOvest.NO_SEDENTARI];
x5NO=[v tNordOvest.NO_ECCESSO_PESO tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE tNordOvest.NO_SEDENTARI tNordOvest.NO_MA_RESPIRATORIE];
YNO=tNordOvest.NO_IPERTENSIONE;
% Creazione grafico crossvalidazione
figure('Name','Crossvalidazione Nord Ovest','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM Nord Ovest')
ylabel('EQM')
xlabel('Numero Regressori')
% eqm cv
regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));
n_regressori = [1, 2, 3, 4, 5];
mse1_NO = crossval('mse',x1NO,YNO,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse2_NO = crossval('mse',x2NO,YNO,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse3_NO = crossval('mse',x3NO,YNO,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse4_NO = crossval('mse',x4NO,YNO,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse5_NO = crossval('mse',x5NO,YNO,'Predfun',regf, 'kfold', 5,'MCReps',1000);
Vettore_mse_NO = [mse1_NO mse2_NO mse3_NO mse4_NO mse5_NO];
    
hold on
plot(n_regressori, Vettore_mse_NO)

%calcoliamo gli R2 dei modelli
[b1NO,bint1NO,res1NO,rint1NO,r1NO] = regress(YNO,x1NO);
[b2NO,bint2NO,res2NO,rint2NO,r2NO] = regress(YNO,x2NO);
[b3NO,bint3NO,res3NO,rint3NO,r3NO] = regress(YNO,x3NO);
[b4NO,bint4NO,res4NO,rint4NO,r4NO] = regress(YNO,x4NO);
[b5NO,bint5NO,res5NO,rint5NO,r5NO] = regress(YNO,x5NO);
Vettore_R2_NO = [r1NO(1) r2NO(1) r3NO(1) r4NO(1) r5NO(1)]
subplot(2,1,2)
plot([1 2 3 4 5],Vettore_R2_NO)
title('Grafico R2 Nord Ovest')
xlabel('Numero Regressori')
ylabel('Valore R2')

hold off

% Modello Completo
NO_lm1 = fitlm(tNordOvest,'ResponseVar','NO_IPERTENSIONE', 'PredictorVars',{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'})
NO_res = NO_lm1.Residuals.Raw;
figure
plot(NO_lm1);

% Verifica ottimalità con GLS
NO_Glm = fitglm(tNordOvest,'ResponseVar','NO_IPERTENSIONE', 'PredictorVars',{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'})

% Verifica non multicollinearità con det(X'X)>0
determinante_NO = det(x3NO'*x3NO)

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
figure
plot(NO_res)
ylabel('Residui');
xlabel('Osservazioni');
yline(nanmean(NO_res), 'Color', 'b', 'LineWidth', 3);
title('Grafico dei residui - Nord Ovest');

% 2. Andamento dei Percentili
figure
qqplot(NO_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati Nord Ovest');

% 3. Incorrelazione dei regressori con i residui
figure
[S,AX,BigAx,H,HAx] = plotmatrix(tNordOvest{:,{'NO_DIABETE','NO_ECCESSO_PESO','NO_MA_ALLERGICHE'}}, NO_res)
title 'Correlazione Residui-Regressori';
AX(1,1).YLabel.String = 'Residui';
AX(1,1).XLabel.String = 'DIABETE';
AX(1,2).XLabel.String = 'ECCESSO DI PESO';
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE';
title('Correlazione Residui-Regressori Nord Ovest')

% Verifica dell'incorrelazione tramite gli indici di correlazione
NO_mat_corr_residui = corrcoef([NO_res, tNordOvest.NO_DIABETE,...
    tNordOvest.NO_ECCESSO_PESO, tNordOvest.NO_MA_ALLERGICHE], 'Rows','complete');
NO_res_corr_w_reg = NO_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

figure
autocorr(NO_res)
xlabel('Lag');
ylabel('Autocorrelazione dei Residui');
title("Autocorrelazione Nord Ovest");

% 4. Ricerca degli outliers
residui_studentizzati = NO_lm1.Residuals.Studentized;
figure
scatter(NO_lm1.Fitted, residui_studentizzati)
xlabel("Fitted data");
ylabel("Residui studentizzati");
yline(3, '--b');
yline(-3, '--b');
title('Residui studentizzati vs Fitted data Nord Ovest');

% 5. Varianza dei residui
figure
plotResiduals(NO_lm1, 'fitted', 'Marker','o')
title('Residuals vs Fitted data Nord Ovest')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(NO_lm1,'exact','both')

% 7. Test di Breusch-Pagan per l'omoschedasticità
pval=TestHet(NO_res,[tNordOvest.NO_ECCESSO_PESO tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% IC beta con bootstrap semi-parametrico (distribuzione errori ignota)
mboot=10000;
beta_boot_NO=nan(mboot,4);
for i=1:mboot
    idx=unidrnd(25,25,1);
    res=NO_res(idx);
    y_boot=x3NO*b3NO+res;
    beta_boot_NO(i,:)=regress(y_boot,x3NO);
end
% distribuzioni beta_boot_NO
figure
subplot(2,2,1)
histfit(beta_boot_NO(:,1));
title('distribuzione intercetta NO');
subplot(2,2,2)
histfit(beta_boot_NO(:,2));
title('distribuzione beta eccesso peso NO');
subplot(2,2,3)
histfit(beta_boot_NO(:,3));
title('distribuzione beta diabete NO');
subplot(2,2,4)
histfit(beta_boot_NO(:,4));
title('distribuzione beta malattie allergiche NO');
%media beta bootstrap
beta_boot_NO_mean=mean(beta_boot_NO);
%varianza beta bootstrap
beta_boot_NO_var=var(beta_boot_NO);
%IC 95% beta bootstrap NO
IC_NO=quantile(beta_boot_NO,[0.025 0.975]);
disp('intercetta NO + IC 95% Bootstrap');
disp([IC_NO(1,1) beta_boot_NO_mean(1) IC_NO(2,1)]);
disp('beta eccesso peso NO + IC 95% Bootstrap');
disp([IC_NO(1,2) beta_boot_NO_mean(2) IC_NO(2,2)]);
disp('beta diabete NO + IC 95% Bootstrap');
disp([IC_NO(1,3) beta_boot_NO_mean(3) IC_NO(2,3)]);
disp('beta malattie allergiche NO + IC 95% Bootstrap');
disp([IC_NO(1,4) beta_boot_NO_mean(4) IC_NO(2,4)]);

%forecast regressione lineare
x = [ones(length(tNordOvest.NO_DIABETE(1:end-5,:)),1) tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
y = tNordOvest.NO_IPERTENSIONE(1:end-5,:);
x_last5 = [ones(length(tNordOvest.NO_DIABETE(end-4:end,:)),1) tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:) tNordOvest.NO_ECCESSO_PESO(end-4:end,:)];
lmNO = fitlm(x,y)
[ypred,yci] = predict(lmNO,x_last5,'alpha',0.05,'Prediction','observation','Simultaneous','on')
err = immse(ypred,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-ypred).^2)
figure
plot(ypred)
hold on
plot(yci,'k--')
plot(tNordOvest.NO_IPERTENSIONE(end-4:end),'r')
legend('previsione','IC 95% lb','IC 95% ub','osservazione')
hold off


%% Regressione dinamica
params = [1 1 1 1];
x_regDin = [tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE tNordOvest.NO_ECCESSO_PESO];
x_regDin = (x_regDin);
y = (tNordOvest.NO_IPERTENSIONE);
funzioneMap = @(params) map(params, x_regDin, NO_lm1.Coefficients.Estimate(1), NO_lm1.Coefficients.Estimate(2), NO_lm1.Coefficients.Estimate(3), NO_lm1.Coefficients.Estimate(4));
modelNO = ssm(funzioneMap)
estModel = estimate(modelNO, y, params)

obs_err = (cell2mat(estModel.D).^2)
sta_err = ((estModel.B).^2)

filterMdl = filter(estModel,y);
alpha_flt = filterMdl(:,1);
beta_flt = filterMdl(:,2:4);
beta_flt1 = filterMdl(:,2);
beta_flt2 = filterMdl(:,3);
beta_flt3 = filterMdl(:,4);

smoothMdl = smooth(estModel,y);
alpha_smo = smoothMdl(:,1);

beta_smo1 = smoothMdl(:,2);
beta_smo2 = smoothMdl(:,3);
beta_smo3 = smoothMdl(:,4);

y3_flt = alpha_flt + (beta_flt1.*x_regDin(:,1)) + (beta_flt2.*x_regDin(:,2)) + (beta_flt3.*x_regDin(:,3));
res = y - y3_flt;
mean(res)
mean_res = mean(res)
kpsstest(res)
figure
subplot(2,2,1)
plot(res)
yline(mean_res)
title('Residuals')
subplot(2,2,2)
histfit(res)
title('Histfit Residuals')
subplot(2,2,3)
autocorr(res)
subplot(2,2,4)
parcorr(res)

figure
plot(y3_flt)
hold on
plot(y)
legend('filter','osservazioni')
hold off

y3_smo = alpha_smo + (beta_smo1.*x_regDin(:,1)) + (beta_smo2.*x_regDin(:,2)) + (beta_smo3.*x_regDin(:,3));
res = y - y3_smo;
mean(res)
mean_res = mean(res)
kpsstest(res)
figure
subplot(2,2,1)
plot(res)
yline(mean_res)
title('Residuals')
subplot(2,2,2)
histfit(res)
title('Histfit Residuals')
subplot(2,2,3)
autocorr(res)
subplot(2,2,4)
parcorr(res)

figure
plot(y3_smo)
hold on
plot(y)
legend('smooth','osservazioni')
hold off

% Previsione un passo in avanti
alpha_flt_forecast = [nan; alpha_flt(1:end-1)];
beta_flt_forecast = [nan nan nan; beta_flt(1:end-1,:)];

beta_flt_forecast1 = beta_flt_forecast(:,1);
beta_flt_forecast2 = beta_flt_forecast(:,2);
beta_flt_forecast3 = beta_flt_forecast(:,3);

y3_frc = alpha_flt_forecast + beta_flt_forecast1.*x_regDin(:,1) + beta_flt_forecast2.*x_regDin(:,2) + beta_flt_forecast3.*x_regDin(:,3);
res = y - y3_frc;
nanmean(res)
mean(res)
mean_res = nanmean(res)
kpsstest(res)
figure
subplot(2,2,1)
plot(res)
yline(mean_res)
title('Residuals')
subplot(2,2,2)
histfit(res)
title('Histfit Residuals')
subplot(2,2,3)
autocorr(res)
subplot(2,2,4)
parcorr(res)

figure
plot(y3_frc)
hold on
plot(y)
hold off

params = [1 1 1 1];
x_regDin = [tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
y = tNordOvest.NO_IPERTENSIONE(1:end-5);
funzioneMap = @(params) map(params, x_regDin, NO_lm1.Coefficients.Estimate(1), NO_lm1.Coefficients.Estimate(2), NO_lm1.Coefficients.Estimate(3), NO_lm1.Coefficients.Estimate(4));
modelNO = ssm(funzioneMap)
[estModel,estParams] = estimate(modelNO, y, params);
x_regDin = [ones(length(tNordOvest.NO_DIABETE(1:end-5,:)),1) tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
x_last5 = [ones(length(tNordOvest.NO_DIABETE(end-4:end,:)),1) tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:) tNordOvest.NO_ECCESSO_PESO(end-4:end,:)];
[yFregDin, yVar] = forecast(estModel,5,y,'Predictors0',x_regDin,'PredictorsF',x_last5,'Beta',estParams)
err = immse(yFregDin,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-yFregDin).^2)
ForecastIntervals(:,1) = yFregDin - 1.96*sqrt(yVar);
ForecastIntervals(:,2) = yFregDin + 1.96*sqrt(yVar);
figure
plot(yFregDin)
hold on
plot(ForecastIntervals,'k--')
plot(tNordOvest.NO_IPERTENSIONE(end-4:end),'r')
legend({'previsione','IC 95% lb','IC 95% ub','osservazione'})
hold off

%% RegArima
%Ciclo per determinare BIC, q e p
x = [tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:)];
y = tNordOvest.NO_IPERTENSIONE(1:end-5,:);
x_last5 = [tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:)];

q_vector = [0 1 2 3 4];
p_vector = [0 1 2 3 4];
Matrix_result = NaN(5,5);

format longg

for p = 0:4
    for q = 0:4
        model = regARIMA(p,0,q);
        try
            estimate_model = estimate(model, y,'X', x);
            res = infer(estimate_model, y, 'X', x);

            bic = summarize(estimate_model);
            Matrix_result(p+1, q+1) = bic.BIC;
            yF = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
            mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-yF).^2);
            Matrix_result2(p+1, q+1) = mse;
        catch
            % Processo non stazionario
            Matrix_result(p+1, q+1) = NaN;
             Matrix_result2(p+1, q+1) = NaN;
        end  
    end
end

figure
subplot(2,1,1)
plot(p_vector, Matrix_result)
legend({'q = 0','q = 1','q = 2','q = 3','q = 4'})
title('Plot BIC rispetto a (p,q)')
xlabel("p");
ylabel("BIC");
hold on
subplot(2,1,2)
plot(p_vector, Matrix_result2)
legend({'q = 0','q = 1','q = 2','q = 3','q = 4'})
title('Plot MSE rispetto a (p,q)')
xlabel("p");
ylabel("MSE");
hold off

% ARMA(0,0,1) modello con migliore rapporto BIC e MSE e con coeff.
% significativi
model = regARIMA(0,0,1);
estimate_model = estimate(model, y,'X', x,'Display','params');
res = infer(estimate_model, y, 'X', x);

[yF,eVar]= forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5)
err = immse(yF,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-yF).^2)

figure
hold on
plot(yF)
plot(tNordOvest.NO_IPERTENSIONE(end-4:end))
legend('previsione','osservazione')
hold off

mean_res = mean(res)
kpsstest(res)
figure
subplot(2,2,1)
plot(res)
yline(mean_res)
title('Residuals')
subplot(2,2,2)
histfit(res)
title('Histfit Residuals')
subplot(2,2,3)
autocorr(res)
subplot(2,2,4)
parcorr(res)

%% OLS per NORDEST
%%CROSS VALIDAZIONE NE
% creazione regressori
v=ones(25,1);
x1NE=[v tNordEst.NE_ECCESSO_PESO];
x2NE=[v tNordEst.NE_ECCESSO_PESO tNordEst.NE_DIABETE];
x3NE=[v tNordEst.NE_ECCESSO_PESO tNordEst.NE_DIABETE tNordEst.NE_MA_ALLERGICHE];
x4NE=[v tNordEst.NE_ECCESSO_PESO tNordEst.NE_DIABETE tNordEst.NE_MA_ALLERGICHE tNordEst.NE_SEDENTARI];
x5NE=[v tNordEst.NE_ECCESSO_PESO tNordEst.NE_DIABETE tNordEst.NE_MA_ALLERGICHE tNordEst.NE_SEDENTARI tNordEst.NE_MA_RESPIRATORIE];
YNE=tNordEst.NE_IPERTENSIONE;

% Creazione grafico crossvalidazione
figure('Name','Crossvalidazione Nord Est','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM Nord Est')
ylabel('EQM')
xlabel('Numero Regressori')
% eqm cv
regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));
n_regressori = [1, 2, 3, 4, 5];
mse1_NE = crossval('mse',x1NE,YNE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse2_NE = crossval('mse',x2NE,YNE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse3_NE = crossval('mse',x3NE,YNE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse4_NE = crossval('mse',x4NE,YNE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse5_NE = crossval('mse',x5NE,YNE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
Vettore_mse_NE = [mse1_NE mse2_NE mse3_NE mse4_NE mse5_NE];
    
hold on
plot(n_regressori, Vettore_mse_NE)

%calcoliamo gli R2 dei modelli
[b1NE,bint1NE,res1NE,rint1NE,r1NE] = regress(YNE,x1NE);
[b2NE,bint2NE,res2NE,rint2NE,r2NE] = regress(YNE,x2NE);
[b3NE,bint3NE,res3NE,rint3NE,r3NE] = regress(YNE,x3NE);
[b4NE,bint4NE,res4NE,rint4NE,r4NE] = regress(YNE,x4NE);
[b5NE,bint5NE,res5NE,rint5NE,r5NE] = regress(YNE,x5NE);
Vettore_R2_NE = [r1NE(1) r2NE(1) r3NE(1) r4NE(1) r5NE(1)]
subplot(2,1,2)
plot([1 2 3 4 5],Vettore_R2_NE)
title('Grafico R2 Nord Est')
xlabel('Numero Regressori')
ylabel('Valore R2')
hold off
% Modello Completo
NE_lm1 = fitlm(tNordEst,'ResponseVar','NE_IPERTENSIONE', 'PredictorVars',{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'})
NE_res = NE_lm1.Residuals.Raw;

figure
plot(NE_lm1);

% Verifica ottimalità con GLS
NE_Glm = fitglm(tNordEst,'ResponseVar','NE_IPERTENSIONE', 'PredictorVars',{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'})

% Verifica non multicollinearità con det(X'X)>0
determinante_NE = det(x3NE'*x3NE)

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
figure
plot(NE_res)
ylabel('Residui');
xlabel('Osservazioni');
yline(nanmean(NE_res), 'Color', 'b', 'LineWidth', 3);
title('Grafico dei residui - Nord Est');

% 2. Andamento dei Percentili
figure
qqplot(NE_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati Nord Est');

% 3. Incorrelazione dei regressori con i residui
figure
[S,AX,BigAx,H,HAx] = plotmatrix(tNordEst{:,{'NE_DIABETE','NE_ECCESSO_PESO','NE_MA_ALLERGICHE'}}, NE_res)
title 'Correlazione Residui-Regressori';
AX(1,1).YLabel.String = 'Residui';
AX(1,1).XLabel.String = 'DIABETE';
AX(1,2).XLabel.String = 'ECCESSO DI PESO';
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE';
title('Correlazione Residui-Regressori Nord Est')

% Verifica dell'incorrelazione tramite gli indici di correlazione
NE_mat_corr_residui = corrcoef([NE_res, tNordEst.NE_DIABETE,...
    tNordEst.NE_ECCESSO_PESO, tNordEst.NE_MA_ALLERGICHE], 'Rows','complete');
NE_res_corr_w_reg = NE_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

figure
autocorr(NE_res)
xlabel('Lag');
ylabel('Autocorrelazione dei Residui');
title("Autocorrelazione Nord Est");

% 4. Ricerca degli outliers
NE_residui_stud = NE_lm1.Residuals.Studentized;
figure
scatter(NE_lm1.Fitted, NE_residui_stud)
xlabel("Fitted data");
ylabel("Residui studentizzati");
yline(3, '--b');
yline(-3, '--b');
title('Residui studentizzati vs Fitted data Nord Est')

% 5. Varianza dei residui
figure
plotResiduals(NE_lm1, 'fitted', 'Marker','o')
title('Residuals vs Fitted data Nord Est')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(NE_lm1,'exact','both')

% 7. Test di Breusch-Pagan per l'omoschedasticità
pval=TestHet(NE_res,[tNordEst.NE_ECCESSO_PESO tNordEst.NE_DIABETE tNordEst.NE_MA_ALLERGICHE], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% IC beta con bootstrap semi-parametrico (distribuzione errori ignota)
mboot=10000;
beta_boot_NE=nan(mboot,4);
for i=1:mboot
    idx=unidrnd(25,25,1);
    res=NE_res(idx);
    y_boot=x3NE*b3NE+res;
    beta_boot_NE(i,:)=regress(y_boot,x3NE);
end
% distribuzioni beta_boot_NE
figure
subplot(2,2,1)
histfit(beta_boot_NE(:,1));
title('distribuzione intercetta NE');
subplot(2,2,2)
histfit(beta_boot_NE(:,2));
title('distribuzione beta eccesso peso NE');
subplot(2,2,3)
histfit(beta_boot_NE(:,3));
title('distribuzione beta diabete NE');
subplot(2,2,4)
histfit(beta_boot_NE(:,4));
title('distribuzione beta malattie allergiche NE');
%media beta bootstrap
beta_boot_NE_mean=mean(beta_boot_NE);
%varianza beta bootstrap
beta_boot_NE_var=var(beta_boot_NE);
%IC 95% beta bootstrap NE
IC_NE=quantile(beta_boot_NE,[0.025 0.975]);
disp('intercetta NE + IC 95% Bootstrap');
disp([IC_NE(1,1) beta_boot_NE_mean(1) IC_NE(2,1)]);
disp('beta eccesso peso NE + IC 95% Bootstrap');
disp([IC_NE(1,2) beta_boot_NE_mean(2) IC_NE(2,2)]);
disp('beta diabete NE + IC 95% Bootstrap');
disp([IC_NE(1,3) beta_boot_NE_mean(3) IC_NE(2,3)]);
disp('beta malattie allergiche NE + IC 95% Bootstrap');
disp([IC_NE(1,4) beta_boot_NE_mean(4) IC_NE(2,4)]);

%% OLS per CENTRO %%
%%CROSS VALIDAZIONE Centro
% creazione regressori
v=ones(25,1);
x1CE=[v tCentro.CE_ECCESSO_PESO];
x2CE=[v tCentro.CE_ECCESSO_PESO tCentro.CE_DIABETE];
x3CE=[v tCentro.CE_ECCESSO_PESO tCentro.CE_DIABETE tCentro.CE_MA_ALLERGICHE];
x4CE=[v tCentro.CE_ECCESSO_PESO tCentro.CE_DIABETE tCentro.CE_MA_ALLERGICHE tCentro.CE_SEDENTARI];
x5CE=[v tCentro.CE_ECCESSO_PESO tCentro.CE_MA_ALLERGICHE tCentro.CE_DIABETE tCentro.CE_SEDENTARI tCentro.CE_MA_RESPIRATORIE];
YCE=tCentro.CE_IPERTENSIONE;
% Creazione grafico crossvalidazione
figure('Name','Crossvalidazione Centro','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM Centro')
ylabel('EQM')
xlabel('Numero Regressori')
% eqm cv
regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));
n_regressori = [1, 2, 3, 4, 5];
mse1_CE = crossval('mse',x1CE,YCE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse2_CE = crossval('mse',x2CE,YCE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse3_CE = crossval('mse',x3CE,YCE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse4_CE = crossval('mse',x4CE,YCE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse5_CE = crossval('mse',x5CE,YCE,'Predfun',regf, 'kfold', 5,'MCReps',1000);
Vettore_mse_CE = [mse1_CE mse2_CE mse3_CE mse4_CE mse5_CE];
    
hold on
plot(n_regressori, Vettore_mse_CE)

%calcoliamo gli R2 dei modelli
[b1CE,bint1CE,res1CE,rint1CE,r1CE] = regress(YCE,x1CE);
[b2CE,bint2CE,resCE,rint2CE,r2CE] = regress(YCE,x2CE);
[b3CE,bint3CE,res3CE,rint3CE,r3CE] = regress(YCE,x3CE);
[b4CE,bint4CE,res4CE,rint4CE,r4CE] = regress(YCE,x4CE);
[b5CE,bint5CE,res5CE,rint5CE,r5CE] = regress(YCE,x5CE);
Vettore_R2_CE = [r1CE(1) r2CE(1) r3CE(1) r4CE(1) r5CE(1)]
subplot(2,1,2)
plot([1 2 3 4 5],Vettore_R2_CE)
title('Grafico R2 Centro')
xlabel('Numero Regressori')
ylabel('Valore R2')
hold off

% Modello Completo
CE_lm1 = fitlm(tCentro,'ResponseVar','CE_IPERTENSIONE', 'PredictorVars',{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'})
CE_res = CE_lm1.Residuals.Raw;

figure
plot(CE_lm1);

% Verifica ottimalità con GLS
CE_Glm = fitglm(tCentro,'ResponseVar','CE_IPERTENSIONE', 'PredictorVars',{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'})

% Verifica non multicollinearità con det(X'X)>0
determinante_CE = det(x3CE'*x3CE)

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
figure
plot(CE_res)
ylabel('Residui');
xlabel('Osservazioni');
yline(nanmean(CE_res), 'Color', 'b', 'LineWidth', 3);
title('Grafico dei residui - Centro');

% 2. Andamento dei Percentili
figure
qqplot(CE_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati Centro');

% 3. Incorrelazione dei regressori con i residui
figure
[S,AX,BigAx,H,HAx] = plotmatrix(tCentro{:,{'CE_DIABETE','CE_ECCESSO_PESO','CE_MA_ALLERGICHE'}}, CE_res)
title 'Correlazione Residui-Regressori';
AX(1,1).YLabel.String = 'Residui';
AX(1,1).XLabel.String = 'DIABETE';
AX(1,2).XLabel.String = 'ECCESSO DI PESO';
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE';
title('Correlazione Residui-Regressori Centro')

% Verifica dell'incorrelazione tramite gli indici di correlazione
CE_mat_corr_residui = corrcoef([CE_res, tCentro.CE_DIABETE,...
    tCentro.CE_ECCESSO_PESO, tCentro.CE_MA_ALLERGICHE], 'Rows','complete');
CE_res_corr_w_reg = CE_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori
 
figure
autocorr(CE_res)
xlabel('Lag');
ylabel('Autocorrelazione dei Residui');
title("Autocorrelazione Centro");

% 4. Ricerca degli outliers
CE_residui_stud = CE_lm1.Residuals.Studentized;
figure
scatter(CE_lm1.Fitted, CE_residui_stud)
xlabel("Fitted data");
ylabel("Residui studentizzati");
yline(3, '--b');
yline(-3, '--b');
title('Residui studentizzati vs Fitted data Centro')

% 5. Varianza dei residui
figure
plotResiduals(CE_lm1, 'fitted', 'Marker','o')
title('Residuals vs Fitted data Centro')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(CE_lm1,'exact','both')

% 7. Test di Breusch-Pagan per l'omoschedasticità
pval=TestHet(CE_res,[tCentro.CE_ECCESSO_PESO tCentro.CE_DIABETE tCentro.CE_MA_ALLERGICHE], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% IC beta con bootstrap semi-parametrico (distribuzione errori ignota)
mboot=10000;
beta_boot_CE=nan(mboot,4);
for i=1:mboot
    idx=unidrnd(25,25,1);
    res=CE_res(idx);
    y_boot=x3CE*b3CE+res;
    beta_boot_CE(i,:)=regress(y_boot,x3CE);
end
% distribuzioni beta_boot_NO
figure
subplot(2,2,1)
histfit(beta_boot_CE(:,1));
title('distribuzione intercetta CE');
subplot(2,2,2)
histfit(beta_boot_CE(:,2));
title('distribuzione beta eccesso peso CE');
subplot(2,2,3)
histfit(beta_boot_CE(:,3));
title('distribuzione beta diabete CE');
subplot(2,2,4)
histfit(beta_boot_CE(:,4));
title('distribuzione beta malattie allergiche CE');
%media beta bootstrap
beta_boot_CE_mean=mean(beta_boot_CE);
%varianza beta bootstrap
beta_boot_CE_var=var(beta_boot_CE);
%IC 95% beta bootstrap CE
IC_CE=quantile(beta_boot_CE,[0.025 0.975]);
disp('intercetta CE + IC 95% Bootstrap');
disp([IC_CE(1,1) beta_boot_CE_mean(1) IC_CE(2,1)]);
disp('beta eccesso peso CE + IC 95% Bootstrap');
disp([IC_CE(1,2) beta_boot_CE_mean(2) IC_CE(2,2)]);
disp('beta diabete CE + IC 95% Bootstrap');
disp([IC_CE(1,3) beta_boot_CE_mean(3) IC_CE(2,3)]);
disp('beta malattie allergiche CE + IC 95% Bootstrap');
disp([IC_CE(1,4) beta_boot_CE_mean(4) IC_CE(2,4)]);

%% OLS per SUD %%
%%CROSS VALIDAZIONE SUD
% creazione regressori
v=ones(25,1);
x1SU=[v tSud.SU_ECCESSO_PESO];
x2SU=[v tSud.SU_ECCESSO_PESO tSud.SU_DIABETE];
x3SU=[v tSud.SU_ECCESSO_PESO tSud.SU_DIABETE tSud.SU_MA_ALLERGICHE];
x4SU=[v tSud.SU_ECCESSO_PESO tSud.SU_DIABETE tSud.SU_MA_ALLERGICHE tSud.SU_SEDENTARI];
x5SU=[v tSud.SU_ECCESSO_PESO tSud.SU_DIABETE tSud.SU_MA_ALLERGICHE tSud.SU_SEDENTARI tSud.SU_MA_RESPIRATORIE];
YSU=tSud.SU_IPERTENSIONE;
% Creazione grafico crossvalidazione
figure('Name','Crossvalidazione Sud','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM Sud')
ylabel('EQM')
xlabel('Numero Regressori')
% eqm cv
regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));
n_regressori = [1, 2, 3, 4, 5];
mse1_SU = crossval('mse',x1SU,YSU,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse2_SU = crossval('mse',x2SU,YSU,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse3_SU = crossval('mse',x3SU,YSU,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse4_SU = crossval('mse',x4SU,YSU,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse5_SU = crossval('mse',x5SU,YSU,'Predfun',regf, 'kfold', 5,'MCReps',1000);
Vettore_mse_SU = [mse1_SU mse2_SU mse3_SU mse4_SU mse5_SU];
    
hold on
plot(n_regressori, Vettore_mse_SU)

%calcoliamo gli R2 dei modelli
[b1SU,bint1SU,res1SU,rint1SU,r1SU] = regress(YSU,x1SU);
[b2SU,bint2SU,res2SU,rint2SU,r2SU] = regress(YSU,x2SU);
[b3SU,bint3SU,res3SU,rint3SU,r3SU] = regress(YSU,x3SU);
[b4SU,bint4SU,res4SU,rint4SU,r4SU] = regress(YSU,x4SU);
[b5SU,bint5SU,res5SU,rint5SU,r5SU] = regress(YSU,x5SU);
Vettore_R2_SU = [r1SU(1) r2SU(1) r3SU(1) r4SU(1) r5SU(1)]
subplot(2,1,2)
plot([1 2 3 4 5],Vettore_R2_SU)
title('Grafico R2 Sud')
xlabel('Numero Regressori')
ylabel('Valore R2')
hold off

% Modello Completo
SU_lm1 = fitlm(tSud,'ResponseVar','SU_IPERTENSIONE', 'PredictorVars',{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'})
SU_res = SU_lm1.Residuals.Raw;

figure
plot(SU_lm1);

% Verifica ottimalità con GLS
SU_Glm = fitglm(tSud,'ResponseVar','SU_IPERTENSIONE', 'PredictorVars',{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'})

% Verifica non multicollinearità con det(X'X)>0
determinante_SU = det(x3SU'*x3SU)

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
figure
plot(SU_res)
ylabel('Residui');
xlabel('Osservazioni');
yline(nanmean(SU_res), 'Color', 'b', 'LineWidth', 3);
title('Grafico dei residui - Sud');

% 2. Andamento dei Percentili
figure
qqplot(SU_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati Sud');

% 3. Incorrelazione dei regressori con i residui
figure
[S,AX,BigAx,H,HAx] = plotmatrix(tSud{:,{'SU_DIABETE','SU_ECCESSO_PESO','SU_MA_ALLERGICHE'}}, SU_res);
title 'Correlazione Residui - Regressori'
AX(1,1).YLabel.String = 'Residui';
AX(1,1).XLabel.String = 'DIABETE';
AX(1,2).XLabel.String = 'ECCESSO DI PESO';
AX(1,3).XLabel.String = 'MALATTIE ALLERGICHE';
title('Correlazione Residui-Regressori Sud')

% Verifica dell'incorrelazione tramite gli indici di correlazione
SU_mat_corr_residui = corrcoef([SU_res, tSud.SU_DIABETE,...
    tSud.SU_ECCESSO_PESO, tSud.SU_MA_ALLERGICHE], 'Rows','complete');
SU_res_corr_w_reg = SU_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

figure
autocorr(SU_res)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')
title("Autocorrelazione Sud")

% 4. Ricerca degli outliers
SU_residui_stud = SU_lm1.Residuals.Studentized;
figure
scatter(SU_lm1.Fitted, SU_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')
title('Residui studentizzati vs Fitted data Sud')

% 5. Varianza dei residui
figure
plotResiduals(SU_lm1, 'fitted', 'Marker','o')
title('Residuals vs Fitted data Sud')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(SU_lm1,'exact','both')

% 7. Test di Breusch-Pagan per l'omoschedasticità
pval=TestHet(SU_res,[tSud.SU_ECCESSO_PESO tSud.SU_DIABETE tSud.SU_MA_ALLERGICHE], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% IC beta con bootstrap semi-parametrico (distribuzione errori ignota)
mboot=10000;
beta_boot_SU=nan(mboot,4);
for i=1:mboot
    idx=unidrnd(25,25,1);
    res=SU_res(idx);
    y_boot=x3SU*b3SU+res;
    beta_boot_SU(i,:)=regress(y_boot,x3SU);
end
% distribuzioni beta_boot_SU
figure
subplot(2,2,1)
histfit(beta_boot_SU(:,1));
title('distribuzione intercetta SU');
subplot(2,2,2)
histfit(beta_boot_SU(:,2));
title('distribuzione beta eccesso peso SU');
subplot(2,2,3)
histfit(beta_boot_SU(:,3));
title('distribuzione beta diabete SU');
subplot(2,2,4)
histfit(beta_boot_SU(:,4));
title('distribuzione beta malattie allergiche SU');
%media beta bootstrap
beta_boot_SU_mean=mean(beta_boot_SU);
%varianza beta bootstrap
beta_boot_SU_var=var(beta_boot_SU);
%IC 95% beta bootstrap SU
IC_SU=quantile(beta_boot_SU,[0.025 0.975]);
disp('intercetta SU + IC 95% Bootstrap');
disp([IC_SU(1,1) beta_boot_SU_mean(1) IC_SU(2,1)]);
disp('beta eccesso peso SU + IC 95% Bootstrap');
disp([IC_SU(1,2) beta_boot_SU_mean(2) IC_SU(2,2)]);
disp('beta diabete SU + IC 95% Bootstrap');
disp([IC_SU(1,3) beta_boot_SU_mean(3) IC_SU(2,3)]);
disp('beta malattie allergiche SU + IC 95% Bootstrap');
disp([IC_SU(1,4) beta_boot_SU_mean(4) IC_SU(2,4)]);

%% OLS per ISOLE %%
%%CROSS VALIDAZIONE ISOLE
% creazione regressori
v=ones(25,1);
x1IS=[v tIsole.IS_DIABETE];
x2IS=[v tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE];
x3IS=[v tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE tIsole.IS_ECCESSO_PESO];
x4IS=[v tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE tIsole.IS_ECCESSO_PESO tIsole.IS_SEDENTARI];
x5IS=[v tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE tIsole.IS_ECCESSO_PESO tIsole.IS_SEDENTARI tIsole.IS_MA_RESPIRATORIE];
YIS=tIsole.IS_IPERTENSIONE;
% Creazione grafico crossvalidazione
figure('Name','Crossvalidazione Isole','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM Isole')
ylabel('EQM')
xlabel('Numero Regressori')
% eqm cv
regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));
n_regressori = [1, 2, 3, 4, 5];
mse1_IS = crossval('mse',x1IS,YIS,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse2_IS = crossval('mse',x2IS,YIS,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse3_IS = crossval('mse',x3IS,YIS,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse4_IS = crossval('mse',x4IS,YIS,'Predfun',regf, 'kfold', 5,'MCReps',1000);
mse5_IS = crossval('mse',x5IS,YIS,'Predfun',regf, 'kfold', 5,'MCReps',1000);
Vettore_mse_IS = [mse1_IS mse2_IS mse3_IS mse4_IS mse5_IS];

hold on
plot(n_regressori, Vettore_mse_IS)

%calcoliamo gli R2 dei modelli
[b1IS,bint1IS,res1IS,rint1IS,r1IS] = regress(YIS,x1IS);
[b2IS,bint2IS,res2IS,rint2IS,r2IS] = regress(YIS,x2IS);
[b3IS,bint3IS,res3IS,rint3IS,r3IS] = regress(YIS,x3IS);
[b4IS,bint4IS,res4IS,rint4IS,r4IS] = regress(YIS,x4IS);
[b5IS,bint5IS,res5IS,rint5IS,r5IS] = regress(YIS,x5IS);
Vettore_R2_IS = [r1IS(1) r2IS(1) r3IS(1) r4IS(1) r5IS(1)]
subplot(2,1,2)
plot([1 2 3 4 5],Vettore_R2_IS)
title('Grafico R2 Isole')
xlabel('Numero Regressori')
ylabel('Valore R2')
hold off
% Modello Completo
IS_lm1 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_ECCESSO_PESO','IS_MA_ALLERGICHE'})

% No eccesso di peso (modello utilizzato)
IS_lm2 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_MA_ALLERGICHE'})
IS_res = IS_lm2.Residuals.Raw;

figure
plot(IS_lm2);

% Verifica ottimalità con GLS
IS_Glm = fitglm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_MA_ALLERGICHE'})

% Verifica non multicollinearità con det(X'X)>0
determinante_IS = det(x2IS'*x2IS)

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
figure
plot(IS_res)
ylabel('Residui');
xlabel('Osservazioni');
yline(nanmean(IS_res), 'Color', 'b', 'LineWidth', 3);
title('Grafico dei residui - Isole');

% 2. Andamento dei Percentili
figure
qqplot(IS_res)
title('Distribuzione Quantili teorici - Quantili residui standardizzati Isole');

% 3. Incorrelazione dei regressori con i residui
figure
[S,AX,BigAx,H,HAx] = plotmatrix(tIsole{:,{'IS_DIABETE','IS_MA_ALLERGICHE'}}, IS_res)
title 'Correlazione Residui - Regressori'
AX(1,1).YLabel.String = 'Residui';
AX(1,1).XLabel.String = 'DIABETE';
AX(1,2).XLabel.String = 'MALATTIE ALLERGICHE';
title('Correlazione Residui-Regressori Isole')

% Verifica dell'incorrelazione tramite gli indici di correlazione
IS_mat_corr_residui = corrcoef([IS_res, tIsole.IS_DIABETE, tIsole.IS_MA_ALLERGICHE], 'Rows','complete');
IS_res_corr_w_reg = IS_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

figure
autocorr(IS_res)
xlabel('Lag');
ylabel('Autocorrelazione dei Residui');
title("Autocorrelazione Isole");

% 4. Ricerca degli outliers
IS_residui_stud = IS_lm2.Residuals.Studentized;
figure
scatter(IS_lm2.Fitted, IS_residui_stud)
xlabel("Fitted data")
ylabel("Residui studentizzati")
yline(3, '--b')
yline(-3, '--b')
title('Residui studentizzati vs Fitted data Isole')

% 5. Varianza dei residui
figure
plotResiduals(IS_lm2, 'fitted', 'Marker','o')
title('Residuals vs Fitted data Isole')

% 6. DW Test per autocorrelazione residui
[p,DW] = dwtest(IS_lm2,'exact','both')

% 7. Test di Breusch-Pagan per l'omoschedasticità
pval=TestHet(IS_res,[tIsole.IS_ECCESSO_PESO tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% IC beta con bootstrap semi-parametrico (distribuzione errori ignota)
mboot=10000;
beta_boot_IS=nan(mboot,3);
for i=1:mboot
    idx=unidrnd(25,25,1);
    res=IS_res(idx);
    y_boot=x2IS*b2IS+res;
    beta_boot_IS(i,:)=regress(y_boot,x2IS);
end
% distribuzioni beta_boot_IS
figure
subplot(2,2,1)
histfit(beta_boot_IS(:,1));
title('distribuzione intercetta IS');
subplot(2,2,2)
histfit(beta_boot_IS(:,2));
title('distribuzione beta diabete IS');
subplot(2,2,3)
histfit(beta_boot_IS(:,3));
title('distribuzione beta malattie allergiche IS');
%media beta bootstrap
beta_boot_IS_mean=mean(beta_boot_IS);
%varianza beta bootstrap
beta_boot_IS_var=var(beta_boot_IS);
%IC 95% beta bootstrap IS
IC_IS=quantile(beta_boot_IS,[0.025 0.975]);
disp('intercetta IS + IC 95% Bootstrap');
disp([IC_IS(1,1) beta_boot_IS_mean(1) IC_IS(2,1)]);
disp('beta diabete IS + IC 95% Bootstrap');
disp([IC_IS(1,2) beta_boot_IS_mean(2) IC_IS(2,2)]);
disp('beta malattie allergiche IS + IC 95% Bootstrap');
disp([IC_IS(1,3) beta_boot_IS_mean(3) IC_IS(2,3)]);

close all