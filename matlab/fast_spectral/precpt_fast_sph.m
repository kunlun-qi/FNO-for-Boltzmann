function y = precpt_fast_sph(N,R,L,Nrho,Nsph,Nsphpre)
% precompute F using Nsphpre (for sigma) points
% rho: Gauss-quadrature
% Spherical Design on whole sphere

tic
F = cell(Nrho,Nsph);

[sph] = getSphericalDesign(Nsph);
g1 = sph.x;
g2 = sph.y;
g3 = sph.z;
[rho,wrho] = lgwt(Nrho,0,R);

for p = 1:Nrho
    for q = 1:Nsph
        F{p,q} = int_F(rho(p),g1(q),g2(q),g3(q),N,L,Nsphpre);
    end
end

filename = ['F_N',num2str(N),'_R',num2str(R),'_L',num2str(L),...
            '_Nrho',num2str(Nrho),'_Nsph',num2str(Nsph),'_Nsphpre',num2str(Nsphpre),'.mat'];
save(filename,'F')

y = 0;

fprintf(1, 'time of precpt_fast_sph is %4.2f sec\n', toc);