function benchmark_spectral_3d()
%BENCHMARK_SPECTRAL_3D Time the same complete f(0)->f(1) task as the FNO.
%
% Precomputation and cache loading are recorded separately.  Online timings
% use resident weights and report one Q evaluation, one four-stage RK4 step,
% and the complete ten-step endpoint map.  Solver console output is captured
% so that printing does not contaminate the measurements.

thisDir    = fileparts(mfilename('fullpath'));
projectDir = fileparts(thisDir);
solverDir  = fullfile(thisDir,'fast_spectral');
addpath(solverDir);
config = jsondecode(fileread(fullfile(projectDir,'config.json')));

resolutions = config.timing.resolutions(:)';
nWarmup = config.timing.spectral_warmup;
nRepeat = config.timing.spectral_repetitions;
rows = repmat(struct(),numel(resolutions),1);

oldDirectory = cd(solverDir);
restoreDirectory = onCleanup(@() cd(oldDirectory));
for index = 1:numel(resolutions)
    N       = resolutions(index);
    S       = config.velocity.S;
    L       = config.velocity.L;
    R       = 2*S;
    Nrho    = config.reference_solver.Nrho;
    Nsph    = config.reference_solver.Nsph;
    Nsphpre = config.reference_solver.Nsphpre;
    dt      = config.reference_solver.dt;
    nSteps  = round(config.reference_solver.time_horizon/dt);
    assert(abs(nSteps*dt-config.reference_solver.time_horizon) < ...
        10*eps(max(1,config.reference_solver.time_horizon)), ...
        'The reference time step must divide the endpoint-map horizon.');
    benchmarkPath = fullfile(projectDir,'data','bkw_benchmark_3d.mat');
    benchmark = load(benchmarkPath,'input_data','physical_time_initial', ...
        'time_horizon','dt','N','L','Nrho','Nsph','Nsphpre');
    assert(benchmark.N == N && benchmark.Nrho == Nrho && ...
        benchmark.Nsph == Nsph && benchmark.Nsphpre == Nsphpre, ...
        'The saved BKW benchmark resolution does not match the timing setup.');
    assert(abs(benchmark.physical_time_initial - ...
        config.bkw_benchmark.physical_time_initial) < 10*eps, ...
        'The saved BKW initial time does not match the timing setup.');
    assert(abs(benchmark.time_horizon - config.reference_solver.time_horizon) ...
        < 10*eps && abs(benchmark.dt-dt) < 10*eps && ...
        abs(benchmark.L-L) < 10*eps, ...
        'The saved BKW benchmark metadata do not match the timing setup.');

    cacheName = ['F_N',num2str(N),'_R',num2str(R),'_L',num2str(L), ...
        '_Nrho',num2str(Nrho),'_Nsph',num2str(Nsph), ...
        '_Nsphpre',num2str(Nsphpre),'.mat'];
    precomputeSeconds = NaN;
    precomputePerformed = false;
    if ~isfile(cacheName)
        precomputeStart = tic;
        precpt_fast_sph(N,R,L,Nrho,Nsph,Nsphpre);
        precomputeSeconds = toc(precomputeStart);
        precomputePerformed = true;
    end
    cacheInfo = dir(cacheName);
    loadStart = tic;
    weights = load(cacheName,'F');
    F = weights.F;
    cacheLoadSeconds = toc(loadStart);
    assert(isequal(size(F),[Nrho,Nsph]), ...
        'The cached fast-spectral weights have incompatible dimensions.');

    % Time the exact frozen benchmark input instead of reconstructing it from
    % a secondary time convention.  The saved tensor layout is
    % [sample, channel, v1, v2, v3], with channel four holding f(t=5.5).
    f0 = double(squeeze(benchmark.input_data(1,4,:,:,:)));
    assert(isequal(size(f0),[N,N,N]) && all(isfinite(f0),'all'), ...
        'The saved BKW benchmark input is invalid.');

    qFunction = @() CBoltz3_fast_sph(f0,N,R,L,Nrho,Nsph,F);
    stepFunction = @() rk4_steps(f0,1,dt,N,R,L,Nrho,Nsph,F);
    mapFunction = @() rk4_steps(f0,nSteps,dt,N,R,L,Nrho,Nsph,F);

    for warmup = 1:nWarmup
        quiet_call(qFunction);
        quiet_call(stepFunction);
        quiet_call(mapFunction);
    end
    qTimes = repeat_timing(qFunction,nRepeat);
    stepTimes = repeat_timing(stepFunction,nRepeat);
    mapTimes = repeat_timing(mapFunction,nRepeat);

    rows(index).N = N;
    rows(index).Nrho = Nrho;
    rows(index).Nsph = Nsph;
    rows(index).Nsphpre = Nsphpre;
    rows(index).S = S;
    rows(index).L = L;
    rows(index).R = R;
    rows(index).dt = dt;
    rows(index).time_horizon = config.reference_solver.time_horizon;
    rows(index).precompute_performed = precomputePerformed;
    rows(index).precompute_seconds = precomputeSeconds;
    rows(index).cache_name = cacheName;
    rows(index).cache_load_seconds = cacheLoadSeconds;
    rows(index).cache_megabytes = cacheInfo.bytes/1024^2;
    rows(index).q_mean_seconds = mean(qTimes);
    rows(index).q_std_seconds = std(qTimes);
    rows(index).rk4_step_mean_seconds = mean(stepTimes);
    rows(index).rk4_step_std_seconds = std(stepTimes);
    rows(index).full_map_mean_seconds = mean(mapTimes);
    rows(index).full_map_std_seconds = std(mapTimes);
    rows(index).warmup = nWarmup;
    rows(index).repetitions = nRepeat;

    fprintf(['N=%d: Q %.4f s, RK4 step %.4f s, full f0->f1 map %.4f s ', ...
        '(%d repetitions).\n'],N,mean(qTimes),mean(stepTimes),mean(mapTimes),nRepeat);
    clear F weights
end

timingTable = struct2table(rows);
csvPath = fullfile(projectDir,'results','spectral_timing.csv');
writetable(timingTable,csvPath);

metadata = struct();
metadata.protocol = 'Complete f(0) to f(1) map; precomputation and cache loading excluded from online time';
metadata.matlab_version = version;
metadata.precision = 'double';
metadata.integrator = 'RK4';
metadata.kernel = config.reference_solver.kernel;
metadata.dt = config.reference_solver.dt;
metadata.time_horizon = config.reference_solver.time_horizon;
metadata.physical_time_initial = config.bkw_benchmark.physical_time_initial;
metadata.physical_time_final = config.bkw_benchmark.physical_time_initial + ...
    config.reference_solver.time_horizon;
metadata.input_source = 'data/bkw_benchmark_3d.mat (frozen exact BKW input)';
metadata.computer_architecture = computer('arch');
metadata.logical_cpu_cores_visible_to_matlab = feature('numcores');
metadata.generated_utc = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
metadata.rows = rows;
jsonPath = fullfile(projectDir,'results','spectral_timing.json');
fileID = fopen(jsonPath,'w');
assert(fileID >= 0,'Unable to open spectral timing JSON for writing.');
cleanupFile = onCleanup(@() fclose(fileID));
fprintf(fileID,'%s',jsonencode(metadata,PrettyPrint=true));
fprintf('Saved %s and %s.\n',csvPath,jsonPath);
end


function values = repeat_timing(functionHandle,nRepeat)
values = zeros(nRepeat,1);
for repetition = 1:nRepeat
    startTime = tic;
    quiet_call(functionHandle);
    values(repetition) = toc(startTime);
end
end


function value = quiet_call(functionHandle) %#ok<INUSD>
[~,value] = evalc('functionHandle()');
end


function f = rk4_steps(f,nSteps,dt,N,R,L,Nrho,Nsph,F)
for step = 1:nSteps
    K1 = CBoltz3_fast_sph(f,N,R,L,Nrho,Nsph,F);
    K2 = CBoltz3_fast_sph(f+0.5*dt*K1,N,R,L,Nrho,Nsph,F);
    K3 = CBoltz3_fast_sph(f+0.5*dt*K2,N,R,L,Nrho,Nsph,F);
    K4 = CBoltz3_fast_sph(f+dt*K3,N,R,L,Nrho,Nsph,F);
    f = f+dt*(K1+2*K2+2*K3+K4)/6;
end
end
