function[A, B, C ,D] = map(paramsInit, x, alpha0, beta0, beta1, beta2)
    A = eye(4);
    B = [sqrt(paramsInit(1)) 0 0 0 ; 0 sqrt(paramsInit(2)) 0 0; 0 0 sqrt(paramsInit(3)) 0 ; 0 0 0 sqrt(paramsInit(4))];
    D = sqrt(paramsInit(3)); % Positive variance constraints
    n = length(x);

    Mean0 = [alpha0; beta0; beta1; beta2];
    Cov0 = eye(4);
    
    StateType = [];
    
    % A = repmat({A},n,1);
    % B = repmat({B},n,1);
    C = num2cell([ones(n,1),x],2);
    D = repmat({D},n,1);
end
