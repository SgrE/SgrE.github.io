%%
% This is the implementation for the toy example in our AISTATS2021 paper entiteld "Kernel regression in high dimensions: Refined
% analysis beyond double descent".


close all;
clear;
rand('state',0);    randn('state',0);

typeker = 'poly'; % or gauss
p = 3;%poly kernel order
barc = 1e-2;% \bar{c} to avoid a large lambda when n is small
varrho =2/3;% decay rate of lambda

NS = [10,20,30,40,50,100,200,300,400,450,480,520,550,600,700,784,900,1000,1200,1500,2000]; % #training data

Iternum =10;% iteration number

%---data generation-----%
ntotal = 5000; train_num = ntotal/2; test_num = ntotal - train_num;
d= 500; dv = 1:d;

 G = randn(ntotal,ntotal);
 [Q,~] = qr(G);
 T = Q(:,1:d);
eigvalpos = diag(1./dv); %hampic decay
%a = 1; eigvalpos = diag(1*dv.^(-2*a)); % poly decay
%a=1; eigvalpos = diag(1*exp(-dv)); % exp decay

X = T*sqrt(eigvalpos);
X = mapstd(X);%normalization
Sigma_d = eigvalpos;

frho = zeros(ntotal,1); %target function
for i = 1:ntotal
    frho(i) = sin(norm(X(i,:)));
end
epsilon = randn(ntotal,1);
Y = frho + epsilon;

X_train = X(1:train_num,:); Y_train = Y(1:train_num);frho_t = frho(1:train_num); epsilon_t = epsilon(1:train_num);
X_test = X(train_num+1:end,:); Y_test = Y(train_num+1:end); frho_te = frho(train_num+1:end);
%----------------%


tau = trace(Sigma_d)/d;

res = zeros(1,length(NS)); res_e = res; res_variance = res; res_variance_e = res; res_bias = res; res_bias_e = res;
res_func = res; res_func_e = res;
for i = 1:length(NS)
    i
    nsample = NS(i);
    
    error = zeros(1,Iternum); variance = error; bias = error; NbX = error;
    for j = 1:Iternum
        pt = randperm(train_num);
        Z_train = X_train(pt(1:nsample),:); ZY_train = Y_train(pt(1:nsample));
        epsilon_tr = epsilon_t(pt(1:nsample)); frho_tr = frho_t(pt(1:nsample));
        
        ntr = size(Z_train,1);
        
        %K_train = create_kernel(Z_train,Z_train,typeker,1/d,p);
        %K_test =  create_kernel(Z_train,X_test,typeker,1/d,p);
        
        if strcmp(typeker,'gauss')
            alpha = exp(-2*tau)*(1+ 2*trace(Sigma_d^2)/d^2); beta = 2*exp(-2*tau); gamma = 1 - (2*tau+1)*exp(-2*tau);
            
            psi = sum(Z_train'.^2)/d - tau; psi = psi';
            A = ones(ntr,1)*psi' + psi*ones(1,ntr);
            TT = -exp(-2*tau)*A + 0.5*exp(-2*tau)*A.*A;
        
            psitest = sum(X_test'.^2)/d - tau; psitest = psitest';
            A_test = psi*ones(1,size(X_test,1)) + ones(ntr,1)*psitest';
            TT_test = -exp(-2*tau)*A_test + 0.5*exp(-2*tau)*A_test.*A_test;
            
            K_train = beta*(Z_train*Z_train')/d + alpha*ones(size(Z_train,1)) + TT;
            K_test =  beta*(Z_train*X_test')/d + alpha*ones(size(Z_train,1),size(X_test,1)) + TT_test;
            
        elseif strcmp(typeker,'poly')
            alpha = 1+p*(p-1)*trace(Sigma_d^2)/2/d^2; beta = p; 
            gamma = (1+tau)^p - 1 - p*tau;
            %gamma = 0; % disentangle the implicit regularization
            K_train = beta*(Z_train*Z_train')/d + alpha*ones(size(Z_train,1));
            K_test =  beta*(Z_train*X_test')/d + alpha*ones(size(Z_train,1),size(X_test,1));
        end
        
        meany = mean(ZY_train);

         tlam = barc*ntr^(-varrho);
         lambda = ntr*tlam; 
         w = (K_train + lambda * eye(size(K_train,1))) \ (ZY_train);
         y_pred = K_test'*w;
        
        error(j) = mean((y_pred - frho_te).^2); %RMSE
        
        temp =  K_test'*((K_train + lambda * eye(size(K_train,1))) \ epsilon_tr); 
        variance(j) = mean(temp.^2); %variance
        temp1 = ( K_test'*((K_train + lambda * eye(size(K_train,1))) \ frho_tr)) - frho_te ;
        bias(j) = mean(temp1.^2); %bias
        
        mean((y_pred - frho_te).^2) - (variance(j) + bias(j))
        
        b = lambda + gamma; %implicit and explicit regularization
        [~,eigX] = eig(K_train); eigXX = diag(eigX);
        NbX(j) = 1/d*sum(eigXX./((b+eigXX).^2));% quality function
    end
    %rank(Z_train*Z_train'/d)
    res(i) = mean(error);
    res_e (i) = std(error);
    
    res_variance(i) = mean(variance); res_variance_e(i) = std(variance);
    res_bias(i) = mean(bias); res_bias_e(i) = std(bias);
    
    res_func(i) = mean(NbX);
    res_func_e(i) = std(NbX);
    %fprintf('%.3f\n',error);
end

%% ---excess risk, vairance----------%
figure(1),errorbar(NS,res,res_e,'-r','linewidth',1.5);grid on;hold on;%expected excess error
errorbar(NS,res_variance,res_variance_e,'--.b','linewidth',1.5);hold on;% expected variance
%--------scaled V1---------%
CC = 1;% scaled V1 for better display
errorbar(NS,CC*res_func,CC*res_func_e,'-.g','linewidth',1.5);hold on;
xlabel('#sample','fontsize',18);ylabel('MSE','fontsize',18);
legend('expected excess error','expected variance E(V)','scaled V_1');

%% ---bias---------------------------%
aaa=1:NS(end);
figure(2),
errorbar(NS,res_bias,res_bias_e,'-m','linewidth',2);hold on;grid on; % bias
xlabel('#sample','fontsize',18);ylabel('MSE','fontsize',18);grid on;hold on;
%-----scaled O(lambda)-------%
CCb = 0.004;%scaled learning rates for better display
plot(CCb*aaa.^(-2*varrho),'-k','linewidth',2);hold on;
legend('bias','O(n^{-2\vartheta})');hold on;