function [ newx, sig ] = amagold( U, gradU, dt, nstep, x, C, mhtest)
sigma = sqrt( 2 * dt * C );
beta = 0.5*C;
p = randn( size(x) );
oldX = x;
oldEnergy = U(x); 
rho = 0;

% do leapfrog
x = x + p * dt/2;
for i = 1 : nstep
    if i > 1
        x = x + p * dt;
    end
    p_old = p;
    gradUx = gradU( x );
    p = ((1 - dt*beta) * p - gradUx * dt + randn(1)*sigma)/(1 + dt * beta);
    rho = rho + gradUx * (p + p_old) * dt / 2;
end
x = x + p * dt/2;

% M-H test
sig = 1;
if mhtest ~= 0
    newEnergy  = U(x);
    if exp(oldEnergy - newEnergy + rho) < rand(1)
        % reject
        x = oldX;
        sig = 0;
    end
end
newx = x;
end