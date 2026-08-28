function generate_external_test_3d()
%GENERATE_EXTERNAL_TEST_3D Create six untouched confirmatory endpoint pairs.
%
% This seed-separated set is generated only after the architecture and C-FNO
% coefficient have been frozen.  It contains two new samples from each of the
% three training families and is never used for fitting, normalization,
% validation, checkpoint selection, or hyperparameter calibration.

thisDir    = fileparts(mfilename('fullpath'));
projectDir = fileparts(thisDir);
solverDir  = fullfile(thisDir,'fast_spectral');
addpath(solverDir);
config = jsondecode(fileread(fullfile(projectDir,'config.json')));

rng(config.external_test.seed,'twister');
N       = config.velocity.N;
S       = config.velocity.S;
L       = config.velocity.L;
R       = 2*S;
Nrho    = config.reference_solver.Nrho;
Nsph    = config.reference_solver.Nsph;
Nsphpre = config.reference_solver.Nsphpre;
dt      = config.reference_solver.dt;
time_horizon = config.reference_solver.time_horizon;
nSteps  = round(time_horizon/dt);
assert(abs(nSteps*dt-time_horizon) < 10*eps(max(1,time_horizon)), ...
    'The reference time step must divide the endpoint-map horizon.');
dv      = 2*L/N;
v       = (-L+dv/2):dv:(L-dv/2);
[v1,v2,v3] = ndgrid(v,v,v);

oldDirectory = cd(solverDir);
restoreDirectory = onCleanup(@() cd(oldDirectory));
cacheName = ['F_N',num2str(N),'_R',num2str(R),'_L',num2str(L), ...
    '_Nrho',num2str(Nrho),'_Nsph',num2str(Nsph), ...
    '_Nsphpre',num2str(Nsphpre),'.mat'];
if ~isfile(cacheName)
    precpt_fast_sph(N,R,L,Nrho,Nsph,Nsphpre);
end
weights = load(cacheName,'F');
F = weights.F;
assert(isequal(size(F),[Nrho,Nsph]), ...
    'The cached fast-spectral weights have incompatible dimensions.');
cacheInfo = dir(cacheName);

nPerFamily = config.external_test.samples_per_family;
nFamilies = 3;
nSamples = nPerFamily*nFamilies;
input_data = zeros(nSamples,4,N,N,N,'single');
target_data = zeros(nSamples,1,N,N,N,'single');
family_id = zeros(nSamples,1,'uint8');
% Columns are tau, center(1:3), sigma, and ten perturbation coefficients.
% Unused entries remain NaN, which makes the precise sampled functions
% recoverable without relying only on the random-number generator version.
family_parameters = nan(nSamples,15);

r2 = v1.^2+v2.^2+v3.^2;
x1 = v1/L;
x2 = v2/L;
x3 = v3/L;
gaussian = @(c,sigma) exp(-((v1-c(1)).^2+(v2-c(2)).^2+ ...
    (v3-c(3)).^2)/(2*sigma^2))/((2*pi)^(3/2)*sigma^3);

sample = 1;
generationStart = tic;
for family = 1:nFamilies
    for member = 1:nPerFamily
        switch family
            case 1
                center = -0.5+rand(1,3);
                sigma = 0.5+0.3*rand();
                f0 = gaussian(center,sigma);
                family_parameters(sample,2:4) = center;
                family_parameters(sample,5) = sigma;

            case 2
                tauMin = -6*log(0.8)+0.05;
                tau = tauMin+(4-tauMin)*rand();
                K = 1-0.5*exp(-tau/6);
                f0 = exp(-r2/(2*K))./(2*(2*pi*K)^(3/2)) .* ...
                    ((5*K-3)/K+(1-K)*r2/(K^2));
                family_parameters(sample,1) = tau;

            case 3
                center = -0.5+rand(1,3);
                sigma = 0.5+0.3*rand();
                M = gaussian(center,sigma);
                coefficients = 2*rand(10,1)-1;
                p = coefficients(1)+coefficients(2)*x1+coefficients(3)*x2+ ...
                    coefficients(4)*x3+coefficients(5)*x1.^2+ ...
                    coefficients(6)*x2.^2+coefficients(7)*x3.^2+ ...
                    coefficients(8)*x1.*x2+coefficients(9)*x1.*x3+ ...
                    coefficients(10)*x2.*x3;
                p = p-sum(M(:).*p(:))/sum(M(:));
                pScale = sqrt(sum(M(:).*p(:).^2)/sum(M(:)));
                assert(pScale > sqrt(eps), ...
                    'Degenerate perturbation in external sample %d.',sample);
                p = tanh(p/pScale);
                f0 = M.*(1+0.2*p);
                family_parameters(sample,2:4) = center;
                family_parameters(sample,5) = sigma;
                family_parameters(sample,6:15) = coefficients.';
        end

        assert(all(isfinite(f0),'all') && min(f0,[],'all') >= 0, ...
            'Invalid external initial condition.');
        f0 = f0/(sum(f0,'all')*dv^3);
        f = f0;
        for step = 1:nSteps
            K1 = CBoltz3_fast_sph(f,N,R,L,Nrho,Nsph,F);
            K2 = CBoltz3_fast_sph(f+0.5*dt*K1,N,R,L,Nrho,Nsph,F);
            K3 = CBoltz3_fast_sph(f+0.5*dt*K2,N,R,L,Nrho,Nsph,F);
            K4 = CBoltz3_fast_sph(f+dt*K3,N,R,L,Nrho,Nsph,F);
            f = f+dt*(K1+2*K2+2*K3+K4)/6;
            assert(all(isfinite(f),'all'), ...
                'Non-finite state in external sample %d at step %d.',sample,step);
        end

        input_data(sample,1,:,:,:) = reshape(single(v1),[1,1,N,N,N]);
        input_data(sample,2,:,:,:) = reshape(single(v2),[1,1,N,N,N]);
        input_data(sample,3,:,:,:) = reshape(single(v3),[1,1,N,N,N]);
        input_data(sample,4,:,:,:) = reshape(single(f0),[1,1,N,N,N]);
        target_data(sample,1,:,:,:) = reshape(single(f),[1,1,N,N,N]);
        family_id(sample) = uint8(family);
        fprintf('External sample %d/%d (family %d, member %d/%d) complete.\n', ...
            sample,nSamples,family,member,nPerFamily);
        sample = sample+1;
    end
end

random_seed = config.external_test.seed;
kernel_name = config.reference_solver.kernel;
cache_name = cacheName;
cache_bytes = cacheInfo.bytes;
matlab_version = version;
generated_utc = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
outputPath = fullfile(projectDir,'data','external_test_data_3V.mat');
save(outputPath,'input_data','target_data','family_id','random_seed', ...
    'family_parameters','time_horizon','dt','dv','v','N','S','L','R', ...
    'Nrho','Nsph','Nsphpre','kernel_name','cache_name','cache_bytes', ...
    'matlab_version','generated_utc','-v7.3');
fprintf('Saved six external test pairs to %s in %.1f s.\n', ...
    outputPath,toc(generationStart));
end
