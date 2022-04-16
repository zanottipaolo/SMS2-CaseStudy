%% TESTHET DEMO [Need stats TB]
%% Load DemoDataset.mat
% These are financial time series of the italian stock and mutual fund 
% market. In the predictors there are the Fama and French factors. 
% Residuals and fitted values are obtained by regressing mutual fund 
% returns on the regressors.
%% Simple Tests
% - Breush-Pagan, Koenker specification test
Pvalues = TestHet(Residuals, Predictors, '-BPK');
% - White test
Pvalues = TestHet(Residuals, Predictors, '-W');
% - White special case test; needs fitted values
Pvalues = TestHet(Residuals, Predictors, '-Ws', FittedValues);