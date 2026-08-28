function Q = CBoltz3_fast_sph(f,N,R,L,Nrho,Nsph,F)
% compute the collision operator Q(f,f) using weak form
% fast spectral method for the classical Boltzmann collision operator
% N # of Fourier modes: f(N,N,N), Q(N,N,N)
% rho: Gauss-quadrature
% Spherical Design on whole sphere

tic
[l1,l2,l3] = ndgrid([0:N/2-1,-N/2:-1]);
FTf = fftn(f);

FQ = zeros(N,N,N);

[sph] = getSphericalDesign(Nsph);
g1 = sph.x;
g2 = sph.y;
g3 = sph.z;
wsph = 4*pi/Nsph;
[rho,wrho] = lgwt(Nrho,0,R);

for p = 1:Nrho
    for q = 1:Nsph
        aa = exp(-1i*pi/L*rho(p)*(l1*g1(q)+l2*g2(q)+l3*g3(q)));      
        bb = F{p,q};
        FQ = FQ + wrho(p)*wsph*rho(p)^2*bb.*fftn(f.*ifftn(aa.*FTf));
    end    
end

Q = real(ifftn(FQ));

fprintf(1, 'time of CBoltz3_fast_sph is %4.2f sec\n', toc);