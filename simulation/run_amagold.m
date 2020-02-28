%% draw probability diagram
xGrid = [-6:xStep:6];
y = exp( - U(xGrid) );
y = y / sum(y) / xStep;
burnin_step = 1000;
%% AMAGOLD
samples = zeros(nsample,1);
x = 0;
succ = 0;
for i = 1:nsample+burnin_step
    [x, sig] = amagold(U, gradU, dt, nstep, x, C, 1);
    if i> burnin_step
      samples(i-burnin_step) = x; 
      succ = succ + sig;
    end
end
save('amagold_doublewell.mat','samples');
