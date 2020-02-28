%% Double-well
%% parameters
clear all;

nsample = 100000;
xStep = 0.1;
C = 0.5;
dt = 0.25;
nstep = 10;

% set random seed
randn('seed',111);

%% set up functions 

U = @(x) (x + 4).*(x + 1).*(x - 1).*(x - 3)/14 + 0.5;
Prob = @(x) exp(-U(x));
gradUPerfect =  @(x) (4*x.^3 + 3*x.^2 - 26*x - 1)/14;
gradU = @(x) gradUPerfect(x) +  randn(1) ;
run_sghmc;
