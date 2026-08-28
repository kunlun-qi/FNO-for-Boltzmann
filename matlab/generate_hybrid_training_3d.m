function generate_hybrid_training_3d()
%GENERATE_HYBRID_TRAINING_3D Replace only the 17 baseline BKW pairs.
%
% The hybrid family contains nine exact BKW states at distinct physical
% times and eight bounded, mass-neutral perturbations of f_BKW(5.5).  Every
% target is produced by the same one-unit fast-spectral RK4 map used by the
% baseline.  The 16 Maxwellian and 17 perturbed-Maxwellian pairs are copied
% without changing a stored value.

thisDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(thisDir);
solverDir = fullfile(thisDir,'fast_spectral');
addpath(solverDir);
config = jsondecode(fileread(fullfile(projectDir,'config.json')));

baselineInputPath = fullfile(projectDir,'data','input_data_3V_baseline50.mat');
baselineTargetPath = fullfile(projectDir,'data','target_data_3V_baseline50.mat');
baselineInput = load(baselineInputPath,'input_data');
baselineTarget = load(baselineTargetPath,'target_data');
input_data = baselineInput.input_data;
target_data = baselineTarget.target_data;
clear baselineInput baselineTarget;

N = config.velocity.N;
S = config.velocity.S;
L = config.velocity.L;
R = 2*S;
Nrho = config.reference_solver.Nrho;
Nsph = config.reference_solver.Nsph;
Nsphpre = config.reference_solver.Nsphpre;
dt = config.reference_solver.dt;
time_horizon = config.reference_solver.time_horizon;
nSteps = round(time_horizon/dt);
assert(abs(nSteps*dt-time_horizon) < 10*eps(max(1,time_horizon)), ...
    'The reference time step must divide the endpoint horizon.');
assert(isequal(size(input_data),[50,4,N,N,N]), ...
    'The baseline input tensor has an unexpected shape.');
assert(isequal(size(target_data),[50,1,N,N,N]), ...
    'The baseline target tensor has an unexpected shape.');

dv = 2*L/N;
v = (-L+dv/2):dv:(L-dv/2);
[v1,v2,v3] = ndgrid(v,v,v);
r2 = v1.^2+v2.^2+v3.^2;
x1 = v1/L;
x2 = v2/L;
x3 = v3/L;

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

exactTimes = reshape(config.hybrid_bkw.exact_time_starts,1,[]);
epsilons = reshape(config.hybrid_bkw.perturbation_epsilons,1,[]);
nominalTime = config.hybrid_bkw.perturbation_nominal_time;
positivityThreshold = config.hybrid_bkw.paper_time_positivity_threshold;
assert(numel(exactTimes) == 9 && numel(epsilons) == 8, ...
    'The hybrid design must contain nine exact and eight perturbed BKW pairs.');
assert(all(exactTimes > positivityThreshold) && nominalTime > positivityThreshold, ...
    'Every selected paper-time BKW state must be nonnegative.');
assert(all(diff(exactTimes) > 0) && all(exactTimes ~= nominalTime), ...
    'Exact training times must be distinct and exclude the benchmark input.');
assert(all(epsilons > 0 & epsilons <= 0.1), ...
    'Perturbation amplitudes must lie in (0,0.1].');

family_id = uint8([ones(16,1);2*ones(17,1);3*ones(17,1)]);
bkw_subtype = zeros(50,1,'uint8'); % 0=non-BKW, 1=exact time, 2=perturbed
bkw_paper_start_time = nan(50,1);
bkw_perturbation_epsilon = nan(50,1);
bkw_perturbation_coefficients = nan(50,10);
mass_initial = nan(50,1);
mass_final = nan(50,1);
minimum_initial = nan(50,1);
minimum_final = nan(50,1);

generationStart = tic;
for localIndex = 1:9
    sample = 16+localIndex;
    paperTime = exactTimes(localIndex);
    f0 = bkw3_paper_time(r2,paperTime);
    f0 = f0/(sum(f0,'all')*dv^3);
    assert(all(isfinite(f0),'all') && min(f0,[],'all') >= 0, ...
        'Invalid exact BKW input at paper time %.8f.',paperTime);
    f = evolve_one_unit(f0,nSteps,dt,N,R,L,Nrho,Nsph,F);

    input_data(sample,4,:,:,:) = reshape(single(f0),[1,1,N,N,N]);
    target_data(sample,1,:,:,:) = reshape(single(f),[1,1,N,N,N]);
    bkw_subtype(sample) = uint8(1);
    bkw_paper_start_time(sample) = paperTime;
    mass_initial(sample) = sum(f0,'all')*dv^3;
    mass_final(sample) = sum(f,'all')*dv^3;
    minimum_initial(sample) = min(f0,[],'all');
    minimum_final(sample) = min(f,[],'all');
    fprintf('Hybrid BKW exact %d/9: t=%.3f -> %.3f complete.\n', ...
        localIndex,paperTime,paperTime+time_horizon);
end

rng(config.hybrid_bkw.perturbation_seed,'twister');
baseBkw = bkw3_paper_time(r2,nominalTime);
baseBkw = baseBkw/(sum(baseBkw,'all')*dv^3);
for localIndex = 1:8
    sample = 25+localIndex;
    coefficients = 2*rand(10,1)-1;
    q = coefficients(1)+coefficients(2)*x1+coefficients(3)*x2+ ...
        coefficients(4)*x3+coefficients(5)*x1.^2+ ...
        coefficients(6)*x2.^2+coefficients(7)*x3.^2+ ...
        coefficients(8)*x1.*x2+coefficients(9)*x1.*x3+ ...
        coefficients(10)*x2.*x3;
    q = q-sum(baseBkw.*q,'all')/sum(baseBkw,'all');
    qScale = sqrt(sum(baseBkw.*q.^2,'all')/sum(baseBkw,'all'));
    assert(qScale > sqrt(eps),'Degenerate BKW perturbation %d.',localIndex);
    q = tanh(q/qScale);
    q = q-sum(baseBkw.*q,'all')/sum(baseBkw,'all');
    q = q/max(abs(q),[],'all');
    weightedMean = sum(baseBkw.*q,'all')/sum(baseBkw,'all');
    assert(abs(weightedMean) < 1e-12 && max(abs(q),[],'all') <= 1+1e-12, ...
        'BKW perturbation %d is not mass-neutral and bounded.',localIndex);

    epsilon = epsilons(localIndex);
    f0 = baseBkw.*(1+epsilon*q);
    assert(all(isfinite(f0),'all') && min(f0,[],'all') >= 0, ...
        'Invalid perturbed BKW input %d.',localIndex);
    f0 = f0/(sum(f0,'all')*dv^3);
    f = evolve_one_unit(f0,nSteps,dt,N,R,L,Nrho,Nsph,F);

    input_data(sample,4,:,:,:) = reshape(single(f0),[1,1,N,N,N]);
    target_data(sample,1,:,:,:) = reshape(single(f),[1,1,N,N,N]);
    bkw_subtype(sample) = uint8(2);
    bkw_paper_start_time(sample) = nominalTime;
    bkw_perturbation_epsilon(sample) = epsilon;
    bkw_perturbation_coefficients(sample,:) = coefficients.';
    mass_initial(sample) = sum(f0,'all')*dv^3;
    mass_final(sample) = sum(f,'all')*dv^3;
    minimum_initial(sample) = min(f0,[],'all');
    minimum_final(sample) = min(f,[],'all');
    fprintf('Hybrid BKW perturbation %d/8: epsilon=%.3f complete.\n', ...
        localIndex,epsilon);
end
generation_seconds = toc(generationStart);

assert(all(isfinite(input_data),'all') && all(isfinite(target_data),'all'), ...
    'The completed hybrid tensors contain non-finite values.');
random_seed = config.seed;
perturbation_seed = config.hybrid_bkw.perturbation_seed;
family_counts = [16 17 17];
hybrid_counts = [9 8];
time_convention = config.bkw_benchmark.time_convention;
time_horizon = config.reference_solver.time_horizon;
kernel_name = config.reference_solver.kernel;
cache_name = cacheName;
cache_bytes = cacheInfo.bytes;
matlab_version = version;
generated_utc = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));

inputPath = fullfile(projectDir,'data','input_data_3V.mat');
targetPath = fullfile(projectDir,'data','target_data_3V.mat');
metadataPath = fullfile(projectDir,'data','hybrid_bkw_metadata.mat');
save(inputPath,'input_data','family_id','family_counts','hybrid_counts', ...
    'bkw_subtype','bkw_paper_start_time','bkw_perturbation_epsilon', ...
    'bkw_perturbation_coefficients','exactTimes','epsilons','nominalTime', ...
    'positivityThreshold','random_seed','perturbation_seed', ...
    'generation_seconds','time_convention','time_horizon','dt','dv','v', ...
    'N','S','L','R','Nrho','Nsph','Nsphpre','kernel_name','cache_name', ...
    'cache_bytes','matlab_version','generated_utc','-v7.3');
save(targetPath,'target_data','family_id','family_counts','hybrid_counts', ...
    'bkw_subtype','bkw_paper_start_time','bkw_perturbation_epsilon', ...
    'generation_seconds','time_convention','time_horizon','dt','dv','N', ...
    'S','L','R','Nrho','Nsph','Nsphpre','kernel_name','cache_name', ...
    'cache_bytes','matlab_version','generated_utc','-v7.3');
save(metadataPath,'family_id','family_counts','hybrid_counts','bkw_subtype', ...
    'bkw_paper_start_time','bkw_perturbation_epsilon', ...
    'bkw_perturbation_coefficients','mass_initial','mass_final', ...
    'minimum_initial','minimum_final','exactTimes','epsilons','nominalTime', ...
    'positivityThreshold','random_seed','perturbation_seed', ...
    'generation_seconds','time_convention','generated_utc','-v7.3');
fprintf('Saved the hybrid 50-pair dataset in %.1f s.\n',generation_seconds);
end


function f = bkw3_paper_time(r2,paperTime)
% Exact positive d=3 BKW family with K(t)=1-exp(-t/6).
K = 1-exp(-paperTime/6);
assert(K >= 3/5,'The selected paper-time BKW state is not nonnegative.');
f = exp(-r2/(2*K))./(2*(2*pi*K)^(3/2)) .* ...
    ((5*K-3)/K+(1-K)*r2/(K^2));
end


function f = evolve_one_unit(f,nSteps,dt,N,R,L,Nrho,Nsph,F)
for step = 1:nSteps
    K1 = CBoltz3_fast_sph(f,N,R,L,Nrho,Nsph,F);
    K2 = CBoltz3_fast_sph(f+0.5*dt*K1,N,R,L,Nrho,Nsph,F);
    K3 = CBoltz3_fast_sph(f+0.5*dt*K2,N,R,L,Nrho,Nsph,F);
    K4 = CBoltz3_fast_sph(f+dt*K3,N,R,L,Nrho,Nsph,F);
    f = f+dt*(K1+2*K2+2*K3+K4)/6;
    assert(all(isfinite(f),'all'), ...
        'Non-finite state at RK4 step %d.',step);
end
end
