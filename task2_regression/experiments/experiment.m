clearvars;
part = 1;
save_folder = '/Users/jonas/cache/sparrkulee';

if part == 1
    cache_folder = '/Users/jonas/cache/bmld_f0_pilot/original/milan';
    results_file = 'part1';
elseif part == 2
    cache_folder = '/Users/jonas/cache/bmld_f0_part2/original/milan';
    results_file = 'part2';
else
    error('Unknown part');
end

null_rs = estimateNull();
analyse(cache_folder, results_file, save_folder);
aggregate(save_folder);
visualise(save_folder, null_rs);

function rs = estimateNull()
    [s, fs] = audioread('/Users/jonas/Library/CloudStorage/Dropbox/EEG/bmld/EEG/stimuli/stories/Milan.wav');
    target_fs = 1024;
    s = resample(s, target_fs, fs);

    % Remove last two samples to match the length of the eeg
    s = s(1:end-2);

    Y = s;
    Y = bsxfun(@minus,Y,sum(Y,1)./size(Y,1));
    Y = Y.*repmat(sqrt(1./max(eps,sum(abs(Y).^2,1))),[size(Y,1),1]);

    rs = zeros(10000, 1);
    for idx = 1:length(rs)
        shift = randi(length(s));
        noise = circshift(s, shift);
        rs(idx) = precached_fast_corr(noise, Y);
    end
end

function aggregate(save_folder)
    files = dir(fullfile(save_folder, 'Binaural*.mat'));
    results = cell(size(files));
    for fileIdx = 1:length(files)
        result = load(fullfile(files(fileIdx).folder, files(fileIdx).name));
        results{fileIdx} = result.corrs;
    end; clear fileIdx;
    save(fullfile(save_folder, 'all_parts'), 'results', '-v7.3');
end

function visualise(save_folder, null_rs)
    d = load(fullfile(save_folder, 'all_parts.mat'));
    corrs = cellfun(@(ss)(ss.s_s), d.results);

    figure(1); clf(1);
    util.boxplot({corrs}, 1);
    hline(prctile(null_rs, 95), {'k:', 'LineWidth', 2});
    hline(prctile(null_rs, 99), {'k:', 'LineWidth', 2});
    text(1.03, prctile(null_rs, 99)*1.3, 'significance level 99%');
    text(1.03, prctile(null_rs, 95)*0.7, 'significance level 95%');
    figIdx = gcf;
    figIdx.CurrentAxes.XTick = [];
    ylabel('Neural tracking (correlation)');
end

function analyse(cache_folder, results_file, save_folder)
    % Set our params
    params.env.fs = 128;
    params.env.startLag = 0;
    params.env.endLag = 0.5;

    params.f0.fs = 1024;
    params.f0.startLag = 0;
    params.f0.endLag = 0.025;

    % Load stimulus, downsample to 1024 Hz
    [s, fs] = audioread('/Users/jonas/Library/CloudStorage/Dropbox/EEG/bmld/EEG/stimuli/stories/Milan.wav');
    target_fs = 1024;
    s = resample(s, target_fs, fs);
    fs = target_fs;

    % Remove last two samples to match the length of the eeg
    s = s(1:end-2);

    % Extract the two features we are interested in (envelope and f0)
    env = abs(s);
    f0 = angle(s);

    files = dir(fullfile(cache_folder, '*.mat'));
    results = cell(size(files));
    for fileIdx = 1:length(files)
        loadPath = fullfile(cache_folder, files(fileIdx).name);

        result = struct;
        result.loadPath = loadPath;

        d = load(loadPath);
        assert(d.fs == fs);
        assert(d.fs == target_fs);
        fs = d.fs;

        reconstructed_env = reconstruct_env(params, d.eeg{1}, env, fs);
        result.reconstructed_env = reconstructed_env;
        result.corrs.env_env = omsi.util.corr(reconstructed_env, env);

        reconstructed_f0 = reconstruct_f0(params, d.eeg{1}, f0, fs);
        result.reconstructed_f0 = reconstructed_f0;
        result.corrs.f0_f0 = omsi.util.corr(reconstructed_f0, f0);
        result.corrs.f0_s = omsi.util.corr(reconstructed_f0, s);

        reconstructed_env = reconstructed_env - prctile(reconstructed_env, 5);
        reconstructed_env(reconstructed_env < 0) = 0;

        reconstructed_f0 = reconstructed_f0 - prctile(reconstructed_f0, 2.5);
        reconstructed_f0(reconstructed_f0 < 0) = 0;
        reconstructed_f0 = reconstructed_f0 ./ prctile(reconstructed_f0, 97.5);
        reconstructed_f0(reconstructed_f0 > 1) = 1;
        reconstructed_f0 = reconstructed_f0 * pi;

        reconstructed_s = real(reconstructed_env .* exp(1i*reconstructed_f0));
        reconstructed_s = reconstructed_s ./ max(abs(reconstructed_s));
        result.reconstructed_s = reconstructed_s;
        result.corrs.s_s = omsi.util.corr(reconstructed_s, s);

        results{fileIdx} = result.corrs;
        save(fullfile(save_folder, files(fileIdx).name), '-struct', 'result', '-v7.3');
        fprintf('%+0.3f %+0.4f %+0.4f %+0.4f\n', result.corrs.env_env, result.corrs.f0_f0, result.corrs.f0_s, result.corrs.s_s);
    end; clear fileIdx;
    save(fullfile(save_folder, results_file), 'results', '-v7.3');
end

function reconstructed_env = reconstruct_env(params, eeg, env, fs)
    % Store length of the original envelope, used for cutting the reconstructed envelope to the correct length
    orig_env_len = length(env);

    % Downsample EEG and env to 64 Hz
    eeg = resample(eeg, params.env.fs, fs);
    env = resample(env, params.env.fs, fs);

    % Reconstruct the envelope
    reconstructed_env = cross_validate(params.env, eeg, env, params.env.fs);

    % Upsample to original sample rate
    reconstructed_env = resample(reconstructed_env, fs, params.env.fs);
    reconstructed_env(end+1:end+fs) = zeros(fs, 1);
    reconstructed_env = reconstructed_env(1:orig_env_len);
end

function reconstructed_f0 = reconstruct_f0(params, eeg, f0, fs)
    % Store length of the original f0, used for cutting the reconstructed f0 to the correct length
    orig_f0_len = length(f0);

    % Filter eeg and f0 between 75 Hz and 200 hz
    eeg = omsi.util.bandpassFilter(eeg, 75, 200, 'fs', fs);
    f0 = omsi.util.bandpassFilter(f0, 75, 200, 'fs', fs);

    % Downsample EEG and f0 to desired frequency
    eeg = resample(eeg, params.f0.fs, fs);
    f0 = resample(f0, params.f0.fs, fs);

    % Reconstruct the envelope
    reconstructed_f0 = cross_validate(params.f0, eeg, f0, params.f0.fs);

    % Upsample to original sample rate
    reconstructed_f0 = resample(reconstructed_f0, fs, params.f0.fs);
    reconstructed_f0(end+1:end+fs) = zeros(fs, 1);
    reconstructed_f0 = reconstructed_f0(1:orig_f0_len);
end

function reconstructedFeature = cross_validate(params, eeg, feature, fs)
    eeg = omsi.util.normalise(eeg);
    feature = omsi.util.normalise(feature);

    windowLen = 40 * fs;
    testStart = 1;
    testEnd = testStart + windowLen - 1;
    reconstructedFeature = nan(size(feature));
    while testEnd <= size(feature, 1)
        testMask = false(size(feature, 1), 1);
        testMask(testStart:testEnd) = true;
        trainMask = ~testMask;

        g = calculateDecoder(params, eeg(trainMask, :), feature(trainMask, :), fs);
        reconstructedFeature(testMask, :) = applyDecoder(params, eeg(testMask, :), g, fs);

        testStart = testEnd + 1;
        testEnd = testStart + windowLen - 1;
    end

    % Add last fold
    testMask = false(size(feature, 1), 1);
    testMask(testStart:end) = true;
    trainMask = ~testMask;
    g = calculateDecoder(params, eeg(trainMask, :), feature(trainMask, :), fs);
    reconstructedFeature(testMask, :) = applyDecoder(params, eeg(testMask, :), g, fs);
end

function g = calculateDecoder(params, eeg, feature, fs)
    eeg = omsi.util.normalise(eeg);
    feature = omsi.util.normalise(feature);

    lags = floor(-params.endLag*fs):ceil(-params.startLag*fs);
    %     lagged_eeg = biopil.backend.mtrf.LagGenerator(eeg, lags);
    lagged_eeg = applyLags(eeg, lags);

    %     Cxx = lagged_eeg' * lagged_eeg;
    Cxx = cov(lagged_eeg);
    Cxy = lagged_eeg' * feature;
    lambda = max(Cxx, [], 'all');
    Cxx = Cxx + lambda * eye(size(Cxx));
    g = Cxx \ Cxy;
end

function reconstructedFeature = applyDecoder(params, eeg, g, fs)
    lags = floor(-params.endLag*fs):ceil(-params.startLag*fs);
    %     lagged_eeg = biopil.backend.mtrf.LagGenerator(eeg, lags);
    lagged_eeg = applyLags(eeg, lags);
    reconstructedFeature = lagged_eeg * g;
end

function lagged_eeg = applyLags(eeg, lags)
    lagged_eeg = zeros(size(eeg, 1), size(eeg, 2) * length(lags), 'single');
    pointerIdx = 1;
    for lagIdx = 1:length(lags)
        lagged_eeg(:, pointerIdx:(pointerIdx+size(eeg, 2)-1)) = circshift(eeg, lags(lagIdx));
        pointerIdx = pointerIdx + size(eeg, 2);
    end
end

function r = precached_fast_corr(X, Y)
    %% Similar to fast_corr but vector Y is not the input signal anymore but already the normalised signal.

    % De-mean Columns:
    X = bsxfun(@minus,X,sum(X,1)./size(X,1)); 
    
    % Normalize by the L2-norm (Euclidean) of Rows:
    X = X.*repmat(sqrt(1./max(eps,sum(abs(X).^2,1))),[size(X,1),1]); 
    
    % Compute Pair-wise Correlation Coefficients:
    r = sum(X.*Y);
end