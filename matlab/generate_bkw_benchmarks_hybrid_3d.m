function generate_bkw_benchmarks_hybrid_3d()
%GENERATE_BKW_BENCHMARKS_HYBRID_3D Exact t=5.5 benchmark and time grid.
%
% Uses the paper clock K(t)=1-exp(-t/6).  For every configured start time,
% the fast spectral method advances one unit and is compared with the exact
% BKW state at t+1.

thisDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(thisDir);
solverDir = fullfile(thisDir,'fast_spectral');
addpath(solverDir);
config = jsondecode(fileread(fullfile(projectDir,'config.json')));

N = config.velocity.N;
S = config.velocity.S;
L = config.velocity.L;
R = 2*S;
Nrho = config.reference_solver.Nrho;
Nsph = config.reference_solver.Nsph;
Nsphpre = config.reference_solver.Nsphpre;
dt = config.reference_solver.dt;
time_horizon = config.bkw_benchmark.time_horizon;
start_times = reshape(config.bkw_benchmark.physical_time_grid,1,[]);
physical_time_initial = config.bkw_benchmark.physical_time_initial;
positivityThreshold = config.hybrid_bkw.paper_time_positivity_threshold;
nSteps = round(time_horizon/dt);
assert(abs(nSteps*dt-time_horizon) < 10*eps(max(1,time_horizon)), ...
    'The configured time step must divide the benchmark horizon.');
assert(start_times(1) == physical_time_initial && ...
    all(start_times > positivityThreshold), ...
    'The benchmark grid must start at the positive exact benchmark time.');

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

dv = 2*L/N;
v = (-L+dv/2):dv:(L-dv/2);
[v1,v2,v3] = ndgrid(v,v,v);
r2 = v1.^2+v2.^2+v3.^2;
nTimes = numel(start_times);
input_grid = zeros(nTimes,4,N,N,N,'single');
target_sm_grid = zeros(nTimes,1,N,N,N,'single');
target_true_grid = zeros(nTimes,1,N,N,N,'single');
spectral_seconds_per_sample = zeros(nTimes,1);

for sample = 1:nTimes
    paperTime = start_times(sample);
    f0 = bkw3_paper_time(r2,paperTime);
    f0 = f0/(sum(f0,'all')*dv^3);
    fTrue = bkw3_paper_time(r2,paperTime+time_horizon);
    fTrue = fTrue/(sum(fTrue,'all')*dv^3);
    fSM = f0;
    solveStart = tic;
    for step = 1:nSteps
        K1 = CBoltz3_fast_sph(fSM,N,R,L,Nrho,Nsph,F);
        K2 = CBoltz3_fast_sph(fSM+0.5*dt*K1,N,R,L,Nrho,Nsph,F);
        K3 = CBoltz3_fast_sph(fSM+0.5*dt*K2,N,R,L,Nrho,Nsph,F);
        K4 = CBoltz3_fast_sph(fSM+dt*K3,N,R,L,Nrho,Nsph,F);
        fSM = fSM+dt*(K1+2*K2+2*K3+K4)/6;
        assert(all(isfinite(fSM),'all'), ...
            'Non-finite BKW benchmark state at sample %d, step %d.',sample,step);
    end
    spectral_seconds_per_sample(sample) = toc(solveStart);

    input_grid(sample,1,:,:,:) = reshape(single(v1),[1,1,N,N,N]);
    input_grid(sample,2,:,:,:) = reshape(single(v2),[1,1,N,N,N]);
    input_grid(sample,3,:,:,:) = reshape(single(v3),[1,1,N,N,N]);
    input_grid(sample,4,:,:,:) = reshape(single(f0),[1,1,N,N,N]);
    target_sm_grid(sample,1,:,:,:) = reshape(single(fSM),[1,1,N,N,N]);
    target_true_grid(sample,1,:,:,:) = reshape(single(fTrue),[1,1,N,N,N]);
    relativeL2 = norm(fSM(:)-fTrue(:))/norm(fTrue(:));
    fprintf('BKW paper time %.2f -> %.2f: SM relative L2 %.3e.\n', ...
        paperTime,paperTime+time_horizon,relativeL2);
end

time_convention = config.bkw_benchmark.time_convention;
kernel_name = config.reference_solver.kernel;
cache_name = cacheName;
cache_bytes = cacheInfo.bytes;
matlab_version = version;
generated_utc = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
gridPath = fullfile(projectDir,'data','bkw_time_grid_3d.mat');
save(gridPath,'input_grid','target_sm_grid','target_true_grid','start_times', ...
    'time_horizon','time_convention','dt','dv','v', ...
    'spectral_seconds_per_sample','N','S','L','R','Nrho','Nsph','Nsphpre', ...
    'kernel_name','cache_name','cache_bytes','matlab_version','generated_utc','-v7.3');

input_data = input_grid(1,:,:,:,:);
target_sm = target_sm_grid(1,:,:,:,:);
target_true = target_true_grid(1,:,:,:,:);
spectralSeconds = spectral_seconds_per_sample(1);
benchmarkPath = fullfile(projectDir,'data','bkw_benchmark_3d.mat');
save(benchmarkPath,'input_data','target_sm','target_true', ...
    'physical_time_initial','time_horizon','time_convention','dt','dv','v', ...
    'spectralSeconds','N','S','L','R','Nrho','Nsph','Nsphpre', ...
    'kernel_name','cache_name','cache_bytes','matlab_version','generated_utc','-v7.3');
fprintf('Saved exact benchmark and %d-time grid.\n',nTimes);
end


function f = bkw3_paper_time(r2,paperTime)
K = 1-exp(-paperTime/6);
assert(K >= 3/5,'The selected paper-time BKW state is not nonnegative.');
f = exp(-r2/(2*K))./(2*(2*pi*K)^(3/2)) .* ...
    ((5*K-3)/K+(1-K)*r2/(K^2));
end
