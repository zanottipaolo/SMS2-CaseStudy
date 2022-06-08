% Owners:
% De Duro Federico      1073477
% Medolago Emanuele     1058907    
% Zanotti Paolo         1074166

rng(6)
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
% Stima dei dati mancanti con media mobile
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
IS_lm1 = fitlm(tIsole,'ResponseVar','IS_IPERTENSIONE', 'PredictorVars',{'IS_DIABETE','IS_MA_ALLERGICHE','IS_ECCESSO_PESO'});
IS_lm =  fitlm([tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_MA_ALLERGICHE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)],tIsole.IS_IPERTENSIONE(1:end-5));

%% forecast regressione lineare
x1 = [tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_MA_ALLERGICHE(1:end-5,:)];
y1 = tIsole.IS_IPERTENSIONE(1:end-5,:);
x_last5 = [tIsole.IS_DIABETE(end-4:end,:) tIsole.IS_MA_ALLERGICHE(end-4:end,:)];
lmIS = fitlm(x1,y1);
[ypred,yci] = predict(lmIS,x_last5,'alpha',0.05,'Prediction','observation','Simultaneous','on');
err = immse(ypred,tIsole.IS_IPERTENSIONE(end-4:end))
mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-ypred).^2);

figure
hold on
plot(T.ANNO(end-4:end), ypred)
plot(T.ANNO(end-4:end), tIsole.IS_IPERTENSIONE(end-4:end))
plot(T.ANNO(end-4:end),yci,'--k')
legend('Previsione','Osservazione','IC lb','IC ub')
title("Confronto Previsione - Osservazione")
xlabel("Anno [Year]",'FontSize', 16)
ylabel("Casi di ipertensione [%]", 'FontSize', 16)
grid()
hold off

%% Regressione dinamica
params = [1 1 1 1];
x_regDin = [tIsole.IS_DIABETE tIsole.IS_MA_ALLERGICHE tIsole.IS_ECCESSO_PESO];
y_regDin = (tIsole.IS_IPERTENSIONE);
funzioneMap = @(params) map(params, x_regDin, IS_lm1.Coefficients.Estimate(1), IS_lm1.Coefficients.Estimate(2), IS_lm1.Coefficients.Estimate(3), IS_lm1.Coefficients.Estimate(4));
modelIS = ssm(funzioneMap)
estModel = estimate(modelIS, y_regDin, params)

obs_err = (cell2mat(estModel.D).^2)
sta_err = ((estModel.B).^2)

filterMdl = filter(estModel,y_regDin);
alpha_flt = filterMdl(:,1);
beta_flt = filterMdl(:,2:4);
beta_flt1 = filterMdl(:,2);
beta_flt2 = filterMdl(:,3);
beta_flt3 = filterMdl(:,4);

smoothMdl = smooth(estModel,y_regDin);
alpha_smo = smoothMdl(:,1);

beta_smo1 = smoothMdl(:,2);
beta_smo2 = smoothMdl(:,3);
beta_smo3 = smoothMdl(:,4);

y3_flt = alpha_flt + (beta_flt1.*x_regDin(:,1)) + (beta_flt2.*x_regDin(:,2)) + (beta_flt3.*x_regDin(:,3));
res = y_regDin - y3_flt;
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
plot(T.ANNO,y3_flt)
hold on
plot(T.ANNO,y_regDin)
legend('filter','osservazioni')
hold off

y3_smo = alpha_smo + (beta_smo1.*x_regDin(:,1)) + (beta_smo2.*x_regDin(:,2)) + (beta_smo3.*x_regDin(:,3));
res = y_regDin - y3_smo;
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
plot(T.ANNO,y3_smo)
hold on
plot(T.ANNO,y_regDin)
legend('smooth','osservazioni')
hold off

% Previsione un passo in avanti
alpha_flt_forecast = [nan; alpha_flt(1:end-1)];
beta_flt_forecast = [nan nan nan; beta_flt(1:end-1,:)];

beta_flt_forecast1 = beta_flt_forecast(:,1);
beta_flt_forecast2 = beta_flt_forecast(:,2);
beta_flt_forecast3 = beta_flt_forecast(:,3);

y3_frc = alpha_flt_forecast + beta_flt_forecast1.*x_regDin(:,1) + beta_flt_forecast2.*x_regDin(:,2) + beta_flt_forecast3.*x_regDin(:,3);
res = y_regDin - y3_frc;
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
plot(T.ANNO,y3_frc)
hold on
plot(T.ANNO,y_regDin)
legend('previsione un passo','osservazioni')
hold off

% Forecast regressione dinamica
params = [1 1 1 1];
x2 = [tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_MA_ALLERGICHE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)];
y2 = tIsole.IS_IPERTENSIONE(1:end-5);
funzioneMap = @(params) map(params, x2, IS_lm.Coefficients.Estimate(1), IS_lm.Coefficients.Estimate(2), IS_lm.Coefficients.Estimate(3), IS_lm.Coefficients.Estimate(4));
modelIS = ssm(funzioneMap)
[estModel,estParams] = estimate(modelIS, y2, params);
x_reg = [ones(length(tIsole.IS_DIABETE(1:end-5,:)),1) tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_MA_ALLERGICHE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)];
x_last5_reg = [ones(length(tIsole.IS_DIABETE(end-4:end,:)),1) tIsole.IS_DIABETE(end-4:end,:) tIsole.IS_MA_ALLERGICHE(end-4:end,:) tIsole.IS_ECCESSO_PESO(end-4:end,:)];
[yFregDin, yVar] = forecast(estModel,5,y2,'Predictors0',x_reg,'PredictorsF',x_last5_reg,'Beta',estParams)
err = immse(yFregDin,tIsole.IS_IPERTENSIONE(end-4:end))
mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yFregDin).^2)
ForecastIntervals(:,1) = yFregDin - 1.96*sqrt(yVar);
ForecastIntervals(:,2) = yFregDin + 1.96*sqrt(yVar);

figure
hold on
plot(T.ANNO(end-4:end), yFregDin)
plot(T.ANNO(end-4:end), tIsole.IS_IPERTENSIONE(end-4:end))
plot(T.ANNO(end-4:end),ForecastIntervals,'--k')
legend('Previsione','Osservazione','IC lb','IC ub')
title("Confronto Previsione - Osservazione")
xlabel("Anno [Year]",'FontSize', 16)
ylabel("Casi di ipertensione [%]", 'FontSize', 16)
grid()
hold off

%% RegArima: tolto intercetta perchè non significativa, diabete e sovrappeso regressori utilizzati,differenzianzione di ordine 1
%Ciclo per determinare BIC, q e p
x = [tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)];
y = tIsole.IS_IPERTENSIONE(1:end-5,:);
x_last5 = [tIsole.IS_DIABETE(end-4:end,:) tIsole.IS_ECCESSO_PESO(end-4:end,:)];

q_vector = [0 1 2 3 4];
p_vector = [0 1 2 3 4];
Matrix_result = NaN(5,5);
Matrix_result2 = NaN(5,5);

format longg
for p = 1:4
    for q = 1:4
           model = regARIMA('ARLags',1:p,'MALags',1:q,'D',1,'Intercept',0);
        try
            estimate_model = estimate(model, y,'X', x);
            res = infer(estimate_model, y, 'X', x);

            bic = summarize(estimate_model);
            Matrix_result(p+1, q+1) = bic.BIC;
            yF = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
            mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yF).^2);
            Matrix_result2(p+1, q+1) = mse;
        catch
            % Processo non stazionario/non invertibile
            Matrix_result(p+1, q+1) = NaN;
            Matrix_result2(p+1, q+1) = NaN;
        end  
    end
end

model = regARIMA('ARLags',1,'D',1,'Intercept',0);
try
    estimate_model = estimate(model, y,'X', x);
    res = infer(estimate_model, y, 'X', x);

    bic = summarize(estimate_model);
    Matrix_result(2,1) = bic.BIC;
    yF = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
    Matrix_result2(2,1) = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yF).^2);
    MSEAR1 = mse;
catch
    % Processo non stazionario/non invertibile
    Matrix_result(2,1) = NaN;
    Matrix_result2(2,1) = NaN;
end  

model = regARIMA('MALags',1,'D',1,'Intercept',0);
try
    estimate_model = estimate(model, y,'X', x);
    res = infer(estimate_model, y, 'X', x);

    bic = summarize(estimate_model);
    Matrix_result(1,2) = bic.BIC;
    yF = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
    mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yF).^2);
    Matrix_result2(1,2) = mse;
catch
    % Processo non stazionario/non invertibile
    Matrix_result(1,2) = NaN;
    Matrix_result2(1,2) = NaN;
end

model = regARIMA('D',1,'Intercept',0);
try
    estimate_model = estimate(model, y,'X', x);
    res = infer(estimate_model, y, 'X', x);

    bic = summarize(estimate_model);
    Matrix_result(1,1) = bic.BIC;
    yF = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
    mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yF).^2);
    Matrix_result2(1,1) = mse;
catch
    % Processo non stazionario/non invertibile
    Matrix_result(1,1) = NaN;
    Matrix_result2(1,1) = NaN;
end

figure
subplot(2,1,1)
plot(p_vector, Matrix_result)
legend({'q = 0','q = 1','q = 2','q = 3','q = 4'})
title('Andamento BIC rispetto a (p,q)', 'FontSize', 16)
xlabel("p", 'FontSize', 16);
ylabel("BIC", 'FontSize', 16);
grid
hold on
subplot(2,1,2)
plot(p_vector, Matrix_result2)
legend({'q = 0','q = 1','q = 2','q = 3','q = 4'})
title('Andamento MSE rispetto a (p,q)', 'FontSize', 16)
xlabel("p", 'FontSize', 16);
ylabel("MSE", 'FontSize', 16);
grid
hold off

% ARIMA(1,1,0) modello con migliore rapporto BIC e MSE, con coefficienti
% significativi, stazionario
model = regARIMA('ARlags',1,'D',1,'Intercept',0);
estimate_model = estimate(model, y,'X', x,'Display','params');
res = infer(estimate_model, y, 'X', x);
estimate_y = y - res;

% Bootstrap semi-parametrico IC coefficienti regArima (i residui, da JB Test 
% via MC, non risultano normali => lasciamo cadere
% l'ipotesi di normalità degli errori per il calcolo degli IC)
n=length(y);
m=200;
y_sim=nan(n,m);
for i=1:m
    idx = unidrnd(20,20,1);
    res_sim = res(idx);
    [yfsim,eVar] = forecast(estimate_model, 20, 'Y0', y(1:2), 'X0', x(1:2,:), 'XF', x);
    y_sim(:,i) = yfsim + res_sim;
end

for j=1:m
    estimate_model_sim = estimate(model, y_sim(:,j),'X', x,'Display','off');
    par_sim_IS(j,1)=estimate_model_sim.Beta(1);
    par_sim_IS(j,2)=estimate_model_sim.Beta(2);
    par_sim_IS(j,3)=cell2mat(estimate_model_sim.AR);
end

figure
subplot(2,2,1)
histfit(par_sim_IS(:,1));
title('distribuzione beta diabete IS');
subplot(2,2,2)
histfit(par_sim_IS(:,2));
title('distribuzione beta sovrappeso IS');
subplot(2,2,3)
histfit(par_sim_IS(:,3));
title('distribuz. coeff. AR IS');
%media beta bootstrap
par_sim_IS_mean=mean(par_sim_IS);
%varianza beta bootstrap
par_sim_IS_var=var(par_sim_IS);
%IC 95% beta bootstrap IS
IC_IS = quantile(par_sim_IS,[0.025 0.975]);
disp('beta diabete IS + IC 95% Bootstrap');
disp([IC_IS(1,1) par_sim_IS_mean(1) IC_IS(2,1)]);
disp('beta sovrappeso IS + IC 95% Bootstrap');
disp([IC_IS(1,2) par_sim_IS_mean(2) IC_IS(2,2)]);
disp('coeff. AR IS + IC 95% Bootstrap');
disp([IC_IS(1,3) par_sim_IS_mean(3) IC_IS(2,3)]);

% forecast regArima
[yF,eVar] = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
err = immse(yF,tIsole.IS_IPERTENSIONE(end-4:end))
mse = mean((tIsole.IS_IPERTENSIONE(end-4:end)-yF).^2);
ForecastInt(:,1) = yF - 1.96*sqrt(eVar);
ForecastInt(:,2) = yF + 1.96*sqrt(eVar);

figure
hold on
plot(T.ANNO(end-4:end), yF)
plot(T.ANNO(end-4:end), tIsole.IS_IPERTENSIONE(end-4:end))
plot(T.ANNO(end-4:end),ForecastInt,'--k')
legend('Previsione','Osservazione','IC lb','IC ub')
title("Confronto Previsione - Osservazione")
xlabel("Anno [Year]",'FontSize', 16)
ylabel("Casi di ipertensione [%]", 'FontSize', 16)
grid()
hold off

% Analisi dei residui
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
parcorr(res,'Method','yule-walker')
% Ricerca degli outliers
dist = (1 / 19) .* ((x - mean(x).^2) / var(x));
estimate_std = sqrt(var(res)) * sqrt(1 - ((1 / 20) + dist));
residui_studentizzati = res ./ estimate_std;

figure
subplot(2,2,1)
    qqplot(res)
    title('Distribuzione Quantili teorici - Quantili residui standardizzati Isole');
subplot(2,2,2)
    [S,AX,BigAx,H,HAx] = plotmatrix([tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)], res)
    title 'Correlazione Residui-Regressori';
    AX(1,1).YLabel.String = 'Residui';
    AX(1,1).XLabel.String = 'DIABETE';
    AX(1,2).XLabel.String = 'SOVRAPPESO';
    title('Correlazione Residui-Regressori Isole')
subplot(2,2,3)
    scatter(estimate_y, res, 'filled')
    title('Residuals vs Fitted data Isole')
subplot(2,2,4)
    scatter(estimate_y, residui_studentizzati)
    xlabel("Fitted data");
    ylabel("Residui studentizzati");
    yline(3, '--b');
    yline(-3, '--b');
    title('Residui studentizzati vs Fitted data Isole');

% Omoschedasticità
pval = TestHet(res,[tIsole.IS_DIABETE(1:end-5,:) tIsole.IS_ECCESSO_PESO(1:end-5,:)], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% Verifica dell'incorrelazione tramite gli indici di correlazione
IS_mat_corr_residui = corrcoef([res, tIsole.IS_DIABETE(1:end-5,:), tIsole.IS_ECCESSO_PESO(1:end-5,:)], 'Rows','complete');
IS_res_corr_w_reg = IS_mat_corr_residui(2:end,1) % Vettore di rho residui - regressori

% T-test media = 0
[h, pval] = ttest(res)
if h == 0
    disp("accetto l'ipotesi nulla, media residui nulla")
else
    disp("rifiuto l'ipotesi nulla, media residui diversa da zero")
end

% test di Lujing Box, residui iid
[h,pValue] = lbqtest(res)
if h == 0
    disp("accetto l'ipotesi nulla, innovazioni iid")
else
    disp("rifiuto l'ipotesi nulla, innovazioni non iid")
end

% Jb Test
x2 = res;
n = length(x2);
JBdata = (skewness(x2).^2)*n/6+((kurtosis(x2)-3).^2)*n/24;

% Simulazione MC
m = 1000;
X0 = randn(m,n);
JB0 = (skewness(X0').^2)*n/6+((kurtosis(X0')-3).^2)*n/24;
alpha = 0.05;
JBcrit = prctile(JB0,100*(1-alpha));
disp(['JBcrit_IS_rA: ',num2str(JBcrit)]);
pval = mean(JB0>JBdata);
stdp = sqrt(pval*(1-pval)/m);
disp(['pvalue_IS_rA: ',num2str(pval)]);
disp(['dev std pvalue_IS_rA: ',num2str(stdp)]);
X1 = chi2rnd(2,m,n);
JB1 = (skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza = mean(JB1>JBcrit);
disp(['potenza test_IS_rA: ',num2str(potenza)]);
% Accetto ipotesi nulla, res normali
