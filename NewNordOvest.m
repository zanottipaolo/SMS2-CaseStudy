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
NO_lm1 = fitlm(tNordOvest,'ResponseVar','NO_IPERTENSIONE', 'PredictorVars',{'NO_DIABETE','NO_MA_ALLERGICHE','NO_ECCESSO_PESO'});

%% forecast regressione lineare
x = [tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
y = tNordOvest.NO_IPERTENSIONE(1:end-5,:);
x_last5 = [tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:) tNordOvest.NO_ECCESSO_PESO(end-4:end,:)];
lmNO = fitlm(x,y);
[ypred,yci] = predict(lmNO,x_last5,'alpha',0.05,'Prediction','observation','Simultaneous','on');
err = immse(ypred,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-ypred).^2);

figure
hold on
plot(T.ANNO(end-4:end), ypred)
plot(T.ANNO(end-4:end), tNordOvest.NO_IPERTENSIONE(end-4:end))
plot(T.ANNO(end-4:end),yci,'--k')
legend('Previsione','Osservazione','IC lb','IC ub')
title("Confronto Previsione - Osservazione")
xlabel("Anno [Year]",'FontSize', 16)
ylabel("Casi di ipertensione [%]", 'FontSize', 16)
grid()
hold off

%% Regressione dinamica
params = [1 1 1 1];
x_regDin = [tNordOvest.NO_DIABETE tNordOvest.NO_MA_ALLERGICHE tNordOvest.NO_ECCESSO_PESO];
y_regDin = (tNordOvest.NO_IPERTENSIONE);
funzioneMap = @(params) map(params, x_regDin, NO_lm1.Coefficients.Estimate(1), NO_lm1.Coefficients.Estimate(2), NO_lm1.Coefficients.Estimate(3), NO_lm1.Coefficients.Estimate(4));
modelNO = ssm(funzioneMap)
estModel = estimate(modelNO, y_regDin, params)

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
x1 = [tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
y1 = tNordOvest.NO_IPERTENSIONE(1:end-5);
funzioneMap = @(params) map(params, x1, lmNO.Coefficients.Estimate(1), lmNO.Coefficients.Estimate(2), lmNO.Coefficients.Estimate(3), lmNO.Coefficients.Estimate(4));
modelNO = ssm(funzioneMap)
[estModel,estParams] = estimate(modelNO, y1, params);
x_reg = [ones(length(tNordOvest.NO_DIABETE(1:end-5,:)),1) tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:) tNordOvest.NO_ECCESSO_PESO(1:end-5,:)];
x_last5_reg = [ones(length(tNordOvest.NO_DIABETE(end-4:end,:)),1) tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:) tNordOvest.NO_ECCESSO_PESO(end-4:end,:)];
[yFregDin, yVar] = forecast(estModel,5,y1,'Predictors0',x_reg,'PredictorsF',x_last5_reg,'Beta',estParams)
err = immse(yFregDin,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-yFregDin).^2)
ForecastIntervals(:,1) = yFregDin - 1.96*sqrt(yVar);
ForecastIntervals(:,2) = yFregDin + 1.96*sqrt(yVar);

figure
hold on
plot(T.ANNO(end-4:end), yFregDin)
plot(T.ANNO(end-4:end), tNordOvest.NO_IPERTENSIONE(end-4:end))
plot(T.ANNO(end-4:end),ForecastIntervals,'--k')
legend('Previsione','Osservazione','IC lb','IC ub')
title("Confronto Previsione - Osservazione")
xlabel("Anno [Year]",'FontSize', 16)
ylabel("Casi di ipertensione [%]", 'FontSize', 16)
grid()
hold off

%% RegArima - Tolto il regressore Eccesso peso perché non significativo
%Ciclo per determinare BIC, q e p
x = [tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:)];
y = tNordOvest.NO_IPERTENSIONE(1:end-5,:);
x_last5 = [tNordOvest.NO_DIABETE(end-4:end,:) tNordOvest.NO_MA_ALLERGICHE(end-4:end,:)];

q_vector = [0 1 2 3 4];
p_vector = [0 1 2 3 4];
Matrix_result = NaN(5,5);
Matrix_result2 = NaN(5,5);

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

% ARIMA(1,0,0) modello con migliore rapporto BIC e MSE, con coefficienti
% significativi, stazionario e invertibile 
model = regARIMA(1,0,0);
estimate_model = estimate(model, y,'X', x,'Display','params');
res = infer(estimate_model, y, 'X', x);
estimate_y = res + y;

% Bootstrap parametrico IC coefficienti regArima (errori normali)
n=length(y);
m=100;
[Y,E_sim,U_sim] = simulate(estimate_model,n,'NumPaths',m,'X',x);
y_sim=nan(n,m);
u = nan(n,m);
for k = 1:m
    E = E_sim(:,k);
    U = U_sim(1,k);
    for j = 2:20
        u(1,k) = U;
        u(j,k) = cell2mat(estimate_model.AR)*u(j-1,k) + E(j);
    end
end

for k=1:m
    for i=1:20
        y_sim(i,k) = estimate_model.Intercept + estimate_model.Beta(1)*x(i,1) + estimate_model.Beta(2)*x(i,2) + u(i,k);
    end
end 

for j=1:m
    estimate_model_sim = estimate(model, y_sim(:,j),'X', x,'Display','off');
    par_sim_NO(j,1)=estimate_model_sim.Intercept;
    par_sim_NO(j,2)=estimate_model_sim.Beta(1);
    par_sim_NO(j,3)=estimate_model_sim.Beta(2);
    par_sim_NO(j,4)=cell2mat(estimate_model_sim.AR);
end

figure
subplot(2,2,1)
histfit(par_sim_NO(:,1));
title('distribuzione intercetta NO');
subplot(2,2,2)
histfit(par_sim_NO(:,2));
title('distribuzione beta diabete NO');
subplot(2,2,3)
histfit(par_sim_NO(:,3));
title('distribuzione beta malattie allergiche NO');
subplot(2,2,4)
histfit(par_sim_NO(:,4));
title('distribuz. coeff. AR NO');
%media beta bootstrap
par_sim_NO_mean=mean(par_sim_NO);
%varianza beta bootstrap
par_sim_NO_var=var(par_sim_NO);
%IC 95% beta bootstrap NE
IC_NO=quantile(par_sim_NO,[0.025 0.975]);
disp('intercetta NO + IC 95% Bootstrap');
disp([IC_NO(1,1) par_sim_NO_mean(1) IC_NO(2,1)]);
disp('beta diabete NO + IC 95% Bootstrap');
disp([IC_NO(1,2) par_sim_NO_mean(2) IC_NO(2,2)]);
disp('beta malattie allergiche NO + IC 95% Bootstrap');
disp([IC_NO(1,3) par_sim_NO_mean(3) IC_NO(2,3)]);
disp('coeff. AR NO + IC 95% Bootstrap');
disp([IC_NO(1,4) par_sim_NO_mean(4) IC_NO(2,4)]);

% forecast regArima
[yF,eVar] = forecast(estimate_model, 5, 'Y0', y, 'X0', x, 'XF', x_last5);
err = immse(yF,tNordOvest.NO_IPERTENSIONE(end-4:end))
mse = mean((tNordOvest.NO_IPERTENSIONE(end-4:end)-yF).^2);
ForecastInt(:,1) = yF - 1.96*sqrt(eVar);
ForecastInt(:,2) = yF + 1.96*sqrt(eVar);

figure
hold on
plot(T.ANNO(end-4:end), yF)
plot(T.ANNO(end-4:end), tNordOvest.NO_IPERTENSIONE(end-4:end))
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
    title('Distribuzione Quantili teorici - Quantili residui standardizzati Nord Ovest');
subplot(2,2,2)
    [S,AX,BigAx,H,HAx] = plotmatrix([tNordOvest.NO_DIABETE(1:end-5,:) tNordOvest.NO_MA_ALLERGICHE(1:end-5,:)], res)
    title 'Correlazione Residui-Regressori';
    AX(1,1).YLabel.String = 'Residui';
    AX(1,1).XLabel.String = 'DIABETE';
    AX(1,2).XLabel.String = 'MALATTIE ALLERGICHE';
    title('Correlazione Residui-Regressori Nord Ovest')
subplot(2,2,3)
    scatter(estimate_y, res, 'filled')
    title('Residuals vs Fitted data Nord Ovest')
subplot(2,2,4)
    scatter(estimate_y, residui_studentizzati)
    xlabel("Fitted data");
    ylabel("Residui studentizzati");
    yline(3, '--b');
    yline(-3, '--b');
    title('Residui studentizzati vs Fitted data Nord Ovest');

% Omoschedasticità
pval = TestHet(res,[tNordOvest.NO_DIABETE(1:end-5,:), tNordOvest.NO_MA_ALLERGICHE(1:end-5,:)], '-BPK')
if pval>0.05
    disp("accetto l'ipotesi nulla, gli errori sono omoschedastici")
else
    disp("rifiuto l'ipotesi nulla, gli errori sono eteroschedastici")
end

% Verifica dell'incorrelazione tramite gli indici di correlazione
NO_mat_corr_residui = corrcoef([res, tNordOvest.NO_DIABETE(1:end-5,:), tNordOvest.NO_MA_ALLERGICHE(1:end-5,:)], 'Rows','complete');
NO_res_corr_w_reg = NO_mat_corr_residui(2:end, 1) % Vettore di rho residui - regressori

% T-test media = 0
[h, pval] = ttest(res)

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
disp(['JBcrit_NO_rA: ',num2str(JBcrit)]);
pval = mean(JB0>JBdata);
stdp = sqrt(pval*(1-pval)/m);
disp(['pvalue_NO_rA: ',num2str(pval)]);
disp(['dev std pvalue_NO_rA: ',num2str(stdp)]);
X1 = chi2rnd(2,m,n);
JB1 = (skewness(X1').^2)*n/6+((kurtosis(X1')-3).^2)*n/24;
potenza = mean(JB1>JBcrit);
disp(['potenza test_NO_rA: ',num2str(potenza)]);
% Accetto ipotesi nulla, res normali
