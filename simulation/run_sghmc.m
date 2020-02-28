%% draw probability diagram
xGrid = [-6:xStep:6];
y = exp( - U(xGrid) );
y = y / sum(y) / xStep;
burnin_step = 1000;
%% SGHMC, no M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample+burnin_step
    x = sghmc(gradU, dt, nstep, x, C);
    if i> burnin_step
      samples(i-burnin_step) = x;
    end 
end
save('sghmc_doublewell.mat','samples');