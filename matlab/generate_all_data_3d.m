function generate_all_data_3d()
%GENERATE_ALL_DATA_3D Reproduce training and evaluation data from scratch.
%
% The first stage reproduces the deterministic base families.  The second
% stage replaces only the 17 BKW members with the positive hybrid design.
% The exact BKW benchmark and the independent six-pair evaluation set are
% generated last.  Precomputed spectral weights are cached locally under
% matlab/fast_spectral and are intentionally excluded from version control.

generate_baseline_training_3d();
generate_hybrid_training_3d();
generate_bkw_benchmarks_hybrid_3d();
generate_external_test_3d();

fprintf(['All 3D Boltzmann endpoint data were generated.  Run ', ...
    'python prepare_hybrid_data.py and then ', ...
    'python scripts/make_data_manifest.py before training.\n']);
end
