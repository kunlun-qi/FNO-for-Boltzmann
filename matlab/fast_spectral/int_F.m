function F = int_F(rho,g1,g2,g3,N,L,Nsphpre)
% precompute F using Nsphpre (for sigma) points
% Spherical Design on whole sphere

[l1,l2,l3] = ndgrid([0:N/2-1,-N/2:-1]);

F = zeros(N,N,N);

[sph] = getSphericalDesign(Nsphpre);
sig1 = sph.x;
sig2 = sph.y;
sig3 = sph.z;
wsph = 4*pi/Nsphpre;

for q = 1:Nsphpre
    % input kernel here
    %B = 1+(g1*sig1(q)+g2*sig2(q)+g3*sig3(q))^2;
    B = 1/(4*pi);
    % integrand
    f = B*(exp(1i*pi/(2*L)*rho*(l1*(g1-sig1(q))+l2*(g2-sig2(q))+l3*(g3-sig3(q))))-1);
    
    F = F+wsph*f;
end