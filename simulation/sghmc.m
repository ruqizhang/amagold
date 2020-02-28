function [ newx ] = sghmc( gradU, dt, nstep, x, C)
p = randn( size(x) );
sigma = sqrt( 2 * dt * C );

for i = 1 : nstep
    p = p - gradU( x ) * dt  - p * C * dt + randn(1)*sigma;
    x = x + p * dt;
end
newx = x;
end
