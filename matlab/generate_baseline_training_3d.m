function generate_baseline_training_3d()
%GENERATE_BASELINE_TRAINING_3D Generate the deterministic 50-pair base data.
% ======================================================================== %
%  0D3V Boltzmann data generation for cutoff Maxwell molecules
%
%  Generates 50 endpoint pairs f(0,v) -> f(1,v) with the repository's
%  fast Fourier spectral collision solver. Intermediate states are not
%  retained.
% ======================================================================== %
thisDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(thisDir);
solverDir = fullfile(thisDir,'fast_spectral');
addpath(solverDir);
config = jsondecode(fileread(fullfile(projectDir,'config.json')));

%% ----------------- Reproducibility ----------------- %%
randomSeed = config.seed;
rng(randomSeed,'twister');

%% ----------------- Velocity / Collision Parameters ----------------- %%
N       = config.velocity.N;      % Grid points per velocity dimension
S       = config.velocity.S;      % Effective support radius of f
R       = 2*S;                    % Relative-velocity truncation radius
L       = (3+sqrt(2))/2*S;        % Anti-aliasing domain: [-L,L)^3
Nrho    = config.reference_solver.Nrho;
Nsph    = config.reference_solver.Nsph;
Nsphpre = config.reference_solver.Nsphpre;

assert(mod(N,2) == 0,'N must be even for CBoltz3_fast_sph.');

dv = 2*L/N;
v  = (-L+dv/2):dv:(L-dv/2);      % Endpoint-free, cell-centered grid
[vv1,vv2,vv3] = ndgrid(v,v,v);

%% ----------------- Precompute / Load Spectral Weights ----------------- %%
% int_F.m specifies the cutoff Maxwell-molecule kernel B = 1/(4*pi).
oldDirectory = cd(solverDir);
restoreDirectory = onCleanup(@() cd(oldDirectory));
precptfilename = ['F_N',num2str(N),'_R',num2str(R),'_L',num2str(L), ...
    '_Nrho',num2str(Nrho),'_Nsph',num2str(Nsph), ...
    '_Nsphpre',num2str(Nsphpre),'.mat'];

if ~isfile(precptfilename)
    fprintf('Precomputing fast-spectral weights: %s\n',precptfilename);
    precpt_fast_sph(N,R,L,Nrho,Nsph,Nsphpre);
end

collisionData = load(precptfilename,'F');
F = collisionData.F;
clear collisionData;
assert(isequal(size(F),[Nrho,Nsph]), ...
    'The cached fast-spectral weights have incompatible dimensions.');

%% ----------------- Endpoint-Pair Parameters ----------------- %%
% Favor the two nonequilibrium families by one sample while keeping the
% three-family allocation as even as possible.
nEachType = config.data.family_counts(:)';
nTypes    = numel(nEachType);
nInit     = sum(nEachType);
assert(nInit == 50,'This generator must produce exactly 50 pairs.');

t0     = 0;
tmax   = config.reference_solver.time_horizon;
dt     = config.reference_solver.dt;
nSteps = round((tmax-t0)/dt);
assert(abs(t0+nSteps*dt-tmax) < 10*eps(max(1,tmax)), ...
    'dt must divide the interval [t0,tmax] exactly.');

% input_data:  [B,4,N,N,N] -> v1, v2, v3, f(0)
% target_data: [B,1,N,N,N] -> f(1)
input_data  = zeros(nInit,4,N,N,N,'single');
target_data = zeros(nInit,1,N,N,N,'single');

coord1 = reshape(single(vv1),[1,1,N,N,N]);
coord2 = reshape(single(vv2),[1,1,N,N,N]);
coord3 = reshape(single(vv3),[1,1,N,N,N]);

massInitial = zeros(nInit,1);
massFinal   = zeros(nInit,1);
minFinal    = zeros(nInit,1);

%% ----------------- Initial-Condition Families ----------------- %%
d = 3;
Gauss3D = @(v1,v2,v3,cx,cy,cz,sig) ...
    exp(-((v1-cx).^2+(v2-cy).^2+(v3-cz).^2)/(2*sig^2)) ...
    ./ ((2*pi)^(3/2)*sig^3);

r2 = vv1.^2 + vv2.^2 + vv3.^2;
x1 = vv1/L;
x2 = vv2/L;
x3 = vv3/L;

% With N=32 and dv approximately 0.345, the original sigma range
% [0.3,0.5] is under-resolved and produces spurious evolution even for an
% exact Maxwellian equilibrium. These widths keep the same Gaussian
% families while resolving their cores on the fixed 32^3 grid.
sigmaMin = 0.5;
sigmaMax = 0.8;

sample_idx = 1;
generationStart = tic;

for init_type = 1:nTypes
    for iInit = 1:nEachType(init_type)
        switch init_type
            case 1
                % Shifted three-dimensional Gaussian (Maxwellian).
                c = -0.5 + rand(1,3);
                sigma = sigmaMin + (sigmaMax-sigmaMin)*rand();
                f0 = Gauss3D(vv1,vv2,vv3,c(1),c(2),c(3),sigma);

            case 2
                % Positive three-dimensional BKW-type distribution.
                % For K = 1-0.5*exp(-tau/6), positivity requires K >= 3/5.
                tauMin = -6*log(0.8) + 0.05;
                tauMax = 4;
                tau = tauMin + (tauMax-tauMin)*rand();
                K = 1 - 0.5*exp(-tau/6);
                f0 = exp(-r2/(2*K)) ./ (2*(2*pi*K)^(d/2)) .* ...
                    (((2+d)*K-d)/K + (1-K)*r2/(K^2));

            case 3
                % Gaussian times a bounded random quadratic perturbation.
                % Maxwellian-weighted scaling keeps the perturbation active
                % where M is appreciable, and tanh guarantees |p| < 1 so
                % that no moment-changing clipping is needed.
                c = -0.5 + rand(1,3);
                sigma = sigmaMin + (sigmaMax-sigmaMin)*rand();
                M = Gauss3D(vv1,vv2,vv3,c(1),c(2),c(3),sigma);

                % [1,x1,x2,x3,x1^2,x2^2,x3^2,x1*x2,x1*x3,x2*x3]
                a = 2*rand(10,1)-1;
                p = a(1) + a(2)*x1 + a(3)*x2 + a(4)*x3 + ...
                    a(5)*x1.^2 + a(6)*x2.^2 + a(7)*x3.^2 + ...
                    a(8)*x1.*x2 + a(9)*x1.*x3 + a(10)*x2.*x3;
                pMean = sum(M(:).*p(:))/sum(M(:));
                p = p-pMean;
                pScale = sqrt(sum(M(:).*p(:).^2)/sum(M(:)));
                assert(pScale > sqrt(eps), ...
                    'Degenerate quadratic perturbation in sample %d.', ...
                    sample_idx);
                p = tanh(p/pScale);
                f0 = M.*(1+0.2*p);
        end

        if any(~isfinite(f0(:))) || min(f0(:)) < 0
            error('Initial condition %d (type %d) is not finite/nonnegative.', ...
                sample_idx,init_type);
        end

        % Correct the negligible finite-domain/quadrature mass bias so that
        % every stored initial condition has unit discrete mass.
        f0 = f0/(sum(f0(:))*dv^3);
        massInitial(sample_idx) = sum(f0(:))*dv^3;

        %% ----------------- RK4 Evolution to t = 1 ----------------- %%
        % Keep the reference solve in double precision and cast only the
        % stored endpoint tensors to single precision.
        f = f0;
        for kt = 1:nSteps
            K1 = CBoltz3_fast_sph(f,N,R,L,Nrho,Nsph,F);
            K2 = CBoltz3_fast_sph(f+0.5*dt*K1,N,R,L,Nrho,Nsph,F);
            K3 = CBoltz3_fast_sph(f+0.5*dt*K2,N,R,L,Nrho,Nsph,F);
            K4 = CBoltz3_fast_sph(f+dt*K3,N,R,L,Nrho,Nsph,F);
            f = f + dt*(K1+2*K2+2*K3+K4)/6;

            if any(~isfinite(f(:)))
                error('Non-finite solution in sample %d at RK4 step %d.', ...
                    sample_idx,kt);
            end
        end

        massFinal(sample_idx) = sum(f(:))*dv^3;
        minFinal(sample_idx)  = min(f(:));

        input_data(sample_idx,1,:,:,:) = coord1;
        input_data(sample_idx,2,:,:,:) = coord2;
        input_data(sample_idx,3,:,:,:) = coord3;
        input_data(sample_idx,4,:,:,:) = ...
            reshape(single(f0),[1,1,N,N,N]);
        target_data(sample_idx,1,:,:,:) = ...
            reshape(single(f),[1,1,N,N,N]);

        fprintf(['Completed %2d/%2d (type %d, member %2d/%2d): ', ...
            'mass drift %.3e, min f(1) %.3e\n'], ...
            sample_idx,nInit,init_type,iInit,nEachType(init_type), ...
            massFinal(sample_idx)-massInitial(sample_idx), ...
            minFinal(sample_idx));
        sample_idx = sample_idx + 1;
    end
end

%% ----------------- Save Only the Requested Endpoint Pairs ----------------- %%
save(fullfile(projectDir,'data','input_data_3V_baseline50.mat'), ...
    'input_data','-v7.3');
save(fullfile(projectDir,'data','target_data_3V_baseline50.mat'), ...
    'target_data','-v7.3');

fprintf(['Generated %d f(0)->f(1) pairs (%s per type) in %.1f s. ', ...
    'Maximum absolute mass drift: %.3e. Minimum target value: %.3e.\n'], ...
    nInit,num2str(nEachType),toc(generationStart), ...
    max(abs(massFinal-massInitial)),min(minFinal));
end
