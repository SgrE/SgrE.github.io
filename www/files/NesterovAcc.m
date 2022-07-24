%% Nesterov's acceleration in Algorithm 1
function  [alpha,F] = NesterovAcc(Ktr,Ytr,ntr, C, eta,tau)
%Input:
% Ktr -- the initial kernel matrix
% Ytr -- the (binary) labels
% ntr -- #training data
% C -- the balance parameter in SVM
% eta -- the regularization parameter for the bounded constraint
% tau -- the regularization parameter for the low rank constraint

%Output:
% alpha -- the optimal dual variable
% F -- the optimal data-adaptive matrix

L = ntr + ntr*3*C^2/4/eta*norm(Ktr,'fro'); %Lipschitz constant. It can be chosen as a smaller one in practice.
espilon = 1e-4;
tmax = 2000;
display = 1;

alpha_0 = randn(ntr,1);
alpha =alpha_0;
if sum(alpha(:)) == 0
    alpha = ones(ntr,1);
end
delta_gradh = zeros(ntr,1);
non_converged = 1;


iter =0;
diff = [];
alphavalue = [];

while (non_converged) && iter <tmax
    
    F = svdthres(alpha,Ktr,Ytr,ntr,eta,tau); % Line 4
    
    alphavalue = [alphavalue,alpha];
    
    labls_alpha = Ytr.*alpha;
    gradh = - ones(ntr,1) + diag(Ytr)*(F.*Ktr)*labls_alpha;  %Line 5
    
    theta_iter = projectionalgo(alpha-gradh/L,Ytr,C.*ones(ntr,1),0); % Line 6
    
    delta_gradh = delta_gradh + 1/2*(iter+1).*gradh;
    beta_iter = projectionalgo(alpha_0-delta_gradh/L,Ytr,C.*ones(ntr,1),0); % Line 7
    
    %update alpha
    alpha = ( (iter+1)*theta_iter + 2*beta_iter )/(iter+3); % Line 8
    
    iter = iter+1;
    if  iter>2
        diff = [diff, norm(alphavalue(:,end)-alphavalue(:,end-1))];
        if display==1
            fprintf('Iter %d   |difference %f |\n', iter,diff(end));
        end
        if  (diff(end)<espilon)
            non_converged =0;
        end
    end
    
end

F = svdthres(alpha,Ktr,Ytr,ntr,eta,tau);

function F = svdthres(alpha,Ktr,Ytr,ntr,eta,tau)
% SVD thresholding operator

Gamma = 1/4/eta*diag(alpha'*diag(Ytr))*Ktr*diag(alpha'*diag(Ytr));
CC = Gamma + ones(ntr); % Gamma(alpha)
CC = (CC+CC')/2;
[u,d,v] = svd(CC);
dd = diag(d)-tau;
F = u*max(0,diag(dd))*v';


function [resb,lambda]=projectionalgo(x,a,c,t)
% Computes the Euclidean projection of x on the set
% 0 <= x_i <= c_i ,  a'*x = t

% Prune out indices for which a_i=0
za=find(abs(a)<1e-10);nza=find(abs(a)>1e-10);
resb(za)=min([c(za)';max([zeros(size(c(za)))';x(za)'])])';
x=x(nza);a=a(nza);c=c(nza);

n=size(x,1);
e=ones(size(x));

anf=[a;+inf];
[veca,inda]=sort([-2*x./a;+inf]);
[vecb,indb]=sort([2*(c-x)./a;+inf]);
asa=anf(inda);asb=anf(indb);

[lastpoint,type]=min([veca(1),vecb(1)]);
grad=t-c(find(a<0))'*a(find(a<0));
gslope=0;
if type==1
    ai=asa(1);
    veca(1)=[];asa(1)=[];
    if ai>=0
        gslope=gslope-ai^2/2;
    else
        gslope=gslope+ai^2/2;
    end
else
    ai=asb(1);
    vecb(1)=[];asb(1)=[];
    if ai>=0
        gslope=gslope+ai^2/2;
    else
        gslope=gslope-ai^2/2;
    end
end


while min([veca(1),vecb(1)])<inf
    [point,type]=min([veca(1),vecb(1)]);
    interval=point-lastpoint;lastpoint=point;
    grad=grad+interval*gslope;
    if grad<0 break; end;
    if type==1
        ai=asa(1);
        veca(1)=[];asa(1)=[];
        if ai>=0
            gslope=gslope-ai^2/2;
        else
            gslope=gslope+ai^2/2;
        end
    else
        ai=asb(1);
        vecb(1)=[];asb(1)=[];
        if ai>=0
            gslope=gslope+ai^2/2;
        else
            gslope=gslope-ai^2/2;
        end
    end;
end
lambda=point-grad/gslope;
res=e;
for i=1:n
    res(i)=x(i)+(lambda*a(i))/2;
    if res(i)<0 res(i)=0; end;
    if res(i)>c(i) res(i)=c(i); end;
end
resb(nza)=res;resb=resb';
