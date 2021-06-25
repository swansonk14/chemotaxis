% References
% Micali: Drift and Behavior of E. coli Cells by Micali et al. - Biophysical Journal 2017 (https://www.sciencedirect.com/science/article/pii/S0006349517310755)
% Taylor: Information and fitness by Taylor, Tishby, and Bialek - arXiv 2007 (https://arxiv.org/abs/0712.4382)
% Clausznitzer: 

%% Set up parameters and equations

close all
clear

exportOptions = struct('Format','eps2',...
    'Color','rgb',...
    'Width',10,...
    'Resolution',300,...
    'FontMode','fixed',...
    'FontSize',10,...
    'LineWidth',2);

% Variables
L = 1000;  % Number of intervals
mi = linspace(0,8,L);  % Methylation levels
ci = logspace(-3,3,L);  % Ligand levels (LOG SPACE)
dm = mean(diff(mi));  % Differences between methylation levels
dc = mean(diff(ci));  % Differences between ligand levels (note: intervals in c are not constant as defined in log space!)

[c, m] = meshgrid(ci,flip(mi));  % Mesh grid of ligand and methylation levels



% Fitness function - estimated

% Parameters - Micali table S1
N = 5;  % Cooperative receptor number (paper uses 13, range [5; 13])
va = 1/3;  % Fraction of Tar receptors (paper uses 1/3)
vs = 2/3;  % Fraction of Tsr receptors (paper uses 1/3)
Kaon = 0.5;  % Active receptors dissociation constant Tar (mM, paper uses 1.0)
Kson = 100000;  % Active receptors dissociation constant Tsr (mM, paper uses 1E6)
Kaoff = 0.02;  % Inactive receptors dissociation constant Tar (mM, paper uses 0.03)
Ksoff = 100;  % Inactive receptors dissociation constant Tsr (mM, paper uses 100)
k = 1;

nu = 0.01;
YT = 9.7; %muM
mT = 1e4;
kyp = 100; % muM^{-1}s^{-1}
kym = nu*kyp;
kzp = 30; % s^{-1}
kzm = nu*kzp;
kRp = 0.0069; %s^{-1}
kRm = nu*kRp;
kBp = 0.12; %s^{-1}
kBm = nu*kBp;

rel_grad = 0.75;

% Free energy and receptor activity (Monod-Wyman-Changeux (MWC) model, Clausznitzer equation 5)
mE = 1 - 0.5 * m;  % Methylation free energy
dF = N * (mE ...
          + va * log((1 + c / Kaoff) ./ (1 + c / Kaon)) ...
          + vs * log((1 + c / Ksoff) ./ (1 + c / Kson)));  % Free energy difference of on and off states
A = 1 ./ (1 + exp(dF));  % Receptor activity



% Drift velocity
vd = k * A .* (1 - A) .* N .* (va .* (c ./ (c + Kaoff) - c ./ (c + Kaon)) + vs .* (c ./ (c + Ksoff) - c ./ (c + Kson))) .* rel_grad;

% Entropy production from phosphorylation and methylation
Yp = (kyp * A + kzm) * YT ./ ((kyp + kym) * A + kzp + kzm);
dSdty = (kyp * (YT - Yp) - kym * Yp) .* A .* log((kyp * (YT - Yp)) ./ (kym * Yp)) + (kzp * Yp - kzm * (YT - Yp)) .* log((kzp * Yp) ./ (kzm * (YT - Yp)));
dSdtm = (kRp - kRm) * (1 - A) * mT * log(kRp / kRm) + (kBp - kBm) * A .^3 * mT * log(kBp / kBm);
EP = (dSdty + dSdtm) / 1000;  % Make values smaller to work

% Choose an output (drift vd or entropy production EP)
% output = EP;
output = vd;

% Input-output curve of methylation and ligand concentration vs output
contourf(log(c), m, output, 64, 'EdgeColor', 'none')
colorbar
colormap(parula)  % winter, parula, eugh, jet, hot
[~, maxi] = max(output);
hold on
plot(log(ci), m(maxi), 'LineWidth', 2 ,'Color', [1 0 0]) %[1 1 1]
xlabel('Ligand c_0');
ylabel('Methylation m')
hold off;

% Distribution of external states of c
r = 0.1;  %2;
Pc = exp(-r * c) ./ sum(exp(-r * c) .* dc);
% figure
% plot(s,Ps)

% Functions to iterate
Eqn2 = @(Pmc) sum(Pmc .* Pc .* dc, 2);  % This should give a vector of Nm by 1

Eqn5 = @(Pm, lambda) exp(lambda * vd) .* Pm ./ sum(Pm .* exp(lambda * vd) .* dm, 1);  % This should give a matrix of Nm by Nc

%% Iterate mutual information algorithm

iimax = 2e4; %10;
etol = 1e-1; %-4, -5
figure
hold on
for lambda = logspace(0, 1, 10)  % (1, 2, 10)
    lambda
    % Pick an initial guess for Pmc and Pm
    Pmi = ones(size(m));
    %dm
    %sum(Pmi.*dm,1)
    Pmi = Pmi ./ sum(Pmi .* dm, 1);
    % 'this is normalised!'
    Pmci = Eqn5(Pmi,lambda);  % columns not normalised!!!!
    for ii = 1:iimax
        %ii
        Pmo = Pmi;
        Pmi = Eqn2(Pmci);  % NaN!!!!
        Pmci = Eqn5(Pmi, lambda);
        %norm(Pmi - Pmo) * dm
        if norm(Pmi - Pmo) * dm <= etol
            disp(['converged for lambda = ' num2str(lambda, 2) ' after ' num2str(ii) ' iterations'])
            Imin = sum(dc * Pc .* sum(dm * Pmci .* log2(eps + Pmci ./ (eps + Pmi)), 1), 2);  % Mutual information Eq. 1
            outmax = sum(dc * Pc .* sum(dm * Pmci .* output, 1), 2);  % Mean fitness from Eq. 4
            
            plot(Imin, outmax, 'x')
            ylabel('mean output')
            xlabel('information I(m;c_0) (bits)')
            break
        end
    end
end

% Conditional probability for final lambda value
%contourf(log(c), m, Pmci, 64, 'EdgeColor', 'none')
%colorbar
%colormap(parula)  % winter, parula, eugh, jet, hot
%ylabel('P(m|c)')
%xlabel('Ligand c')

%% Export plot

%ylim([0 0.01])
%xlim([0 2])
%set(gcf, 'PaperUnits', 'centimeters')
%filename = ['informationfitnesscurve'];
%exportfig(gcf, [filename '.eps'], exportOptions);
%system(['epstopdf ' filename '.eps']);
%system(['rm ' filename '.eps']);