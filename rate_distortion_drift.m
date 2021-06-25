% Based on: Information and fitness, Taylor, Tishby, Bialek et al. - arXiv 2007
% log-space distributed logspace and dc=mean(diff(ci))
% replace v_d by EP (entropy production rate)
close all
clear

exportOptions = struct('Format','eps2',...
    'Color','rgb',...
    'Width',10,...
    'Resolution',300,...
    'FontMode','fixed',...
    'FontSize',10,...
    'LineWidth',2);

% variables
L=1000 %1000
mi = linspace(0,8,L);%1e3);
ci = logspace(-3,3,L);%1e3); % LOG SPACE
dc = mean(diff(ci)); % But intervals in c are not constant as defined in log space!
dm = mean(diff(mi));

[c, m] = meshgrid(ci,flip(mi));

% fitness function - estimated

N=5;
va=1/3;
vs=2/3;
Kaoff=0.02;
Kaon=0.5;
Ksoff=100;
Kson=100000;
k=1;

nu=0.01;
YT=9.7; %muM
mT=1e4;
kyp=100; % muM^{-1}s^{-1}
kym=nu*kyp;
kzp=30; % s^{-1}
kzm=nu*kzp;
kRp=0.0069; %s^{-1}
kRm=nu*kRp;
kBp=0.12; %s^{-1}
kBm=nu*kBp;

rel_grad=0.75;

% free energy dF and receptor activity A
dF=N*(1-0.5*m+va*log((1+c/Kaoff)./(1+c/Kaon))+vs*log((1+c/Ksoff)./(1+c/Kson)));
A=1./(1+exp(dF));

% drift velocity vd
vd=k*A.*(1-A).*N.*(va.*(c./(c+Kaoff)-c./(c+Kaon))+vs.*(c./(c+Ksoff)-c./(c+Kson))).*rel_grad;

% entropy production from phosphorylation and methylation
Yp=(kyp*A+kzm)*YT./((kyp+kym)*A+kzp+kzm);
dSdty=(kyp*(YT-Yp)-kym*Yp).*A.*log((kyp*(YT-Yp))./(kym*Yp))+(kzp*Yp-kzm*(YT-Yp)).*log((kzp*Yp)./(kzm*(YT-Yp)));
dSdtm=(kRp-kRm)*(1-A)*mT*log(kRp/kRm)+(kBp-kBm)*A.^3*mT*log(kBp/kBm);
EP=(dSdty+dSdtm)/1000; % make values smaller to work

% choose an output (drift vd or entropy production EP)
% output=EP;
output=vd;

% input-output curve of methylation and ligand concentration vs output
contourf(log(c),m,output,64,'EdgeColor','none')
colorbar
colormap(parula) % winter, parula, eugh, jet, hot
[~,maxi] = max(output);
hold on
plot(log(ci),m(maxi),'LineWidth',2,'Color',[1 0 0]) %[1 1 1]
xlabel('Ligand c_0');
ylabel('Methylation m')
hold off;

% distribution of external states of c
r=0.1;%2;
Pc = exp(-r*c)./sum(exp(-r*c).*dc);
% figure
% plot(s,Ps)

% functions to iterate
Eqn2 = @(Pmc) sum(Pmc.*Pc.*dc,2); % this should give a vector of Nm by 1

Eqn5 = @(Pm,lambda) exp(lambda*vd).*Pm./sum(Pm.*exp(lambda*vd).*dm,1); % this should give a matrix of Nm by Nc

%% iterate algorithm
iimax = 2e4; %10;
etol = 1e-1; %-4, -5
figure
hold on
for lambda = logspace(0,1,10)%(1,2,10)
    lambda
    % pick an initial guess for Pmc and Pm
    Pmi = ones(size(m));
    %dm
    %sum(Pmi.*dm,1)
    Pmi = Pmi./sum(Pmi.*dm,1);
    %'this is normalised!'
    Pmci = Eqn5(Pmi,lambda); % columns not normalised!!!!
    for ii = 1:iimax
        %ii
        Pmo = Pmi;
        Pmi = Eqn2(Pmci); % NaN!!!!
        Pmci = Eqn5(Pmi,lambda);
        %norm(Pmi - Pmo)*dm
        if norm(Pmi - Pmo)*dm<=etol
            disp(['converged for lambda = ' num2str(lambda,2) ' after ' num2str(ii) ' iterations'])
            Imin = sum(dc*Pc.*sum(dm*Pmci.*log2(eps+Pmci./(eps+Pmi)),1),2); % Mutual information Eq. 1
            outmax = sum(dc*Pc.*sum(dm*Pmci.*output,1),2); % mean fitness from Eq. 4
            
            plot(Imin,outmax,'x')
            ylabel('mean output')
            xlabel('information I(m;c_0) (bits)')
            break
        end
    end
end
% conditional probability for final lambda value
%contourf(log(c),m,Pmci,64,'EdgeColor','none')
%colorbar
%colormap(parula) % winter, parula, eugh, jet, hot
%ylabel('P(m|c)')
%xlabel('Ligand c')

%% export plot
%ylim([0 0.01])
%xlim([0 2])
%set(gcf,'PaperUnits','centimeters')
%filename = ['informationfitnesscurve'];
%exportfig(gcf,[filename '.eps'],exportOptions);
%system(['epstopdf ' filename '.eps']);
%system(['rm ' filename '.eps']);