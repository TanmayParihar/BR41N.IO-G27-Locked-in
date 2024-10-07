% Create the output folder if it doesn't exist
outputFolder = 'G:\locked in\Preprocessed_data';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% List all .mat files in the Data folder
dataFolder = 'G:\locked in\Data';
dataFiles = dir(fullfile(dataFolder, '*.mat'));

% Pre-processing parameters
lowCutoff = 0.1; % Low cutoff frequency for band-pass filter (Hz)
highCutoff = 30; % High cutoff frequency for band-pass filter (Hz)
epochWindow = [-0.2, 0.8]; % Epoch window in seconds (e.g., 200 ms pre-stimulus to 800 ms post-stimulus)

% Power line frequency (adjust according to your region)
powerLineFreq = 50; % Use 60 for regions with 60 Hz power line frequency

for i = 1:length(dataFiles)
    % Load the .mat file
    dataFilePath = fullfile(dataFolder, dataFiles(i).name);
    data = load(dataFilePath);
    
    % Extract the variables from the file
    fs = data.fs;        % Sampling frequency
    trig = data.trig;    % Trigger data
    y = data.y;          % EEG data (samples x 8 channels)
    
    % 1. Band-pass filter the EEG data (0.1 Hz to 30 Hz)
    [b_bp, a_bp] = butter(4, [lowCutoff highCutoff] / (fs / 2), 'bandpass');
    yFiltered = filtfilt(b_bp, a_bp, y);
    
    % 2. Power Line Noise Removal (Notch Filter at 50/60 Hz)
    wo = powerLineFreq / (fs / 2);  % Normalized frequency
    bw = wo / 35;                   % Bandwidth
    [b_notch, a_notch] = iirnotch(wo, bw);
    yFiltered = filtfilt(b_notch, a_notch, yFiltered);
    
    % 3. Muscle Movement Artifacts Removal using ICA
    % Transpose data to channels x samples for ICA
    yICAInput = yFiltered';
    % Perform ICA
    [icasig, A, W] = fastica(yICAInput, 'verbose', 'off');
    
    % Identify artifact components (components with high-frequency content)
    numComponents = size(icasig, 1);
    artifactComponents = [];
    for comp = 1:numComponents
        % Compute power spectral density
        [pxx, f] = pwelch(icasig(comp, :), [], [], [], fs);
        % Find power in muscle artifact frequency range (20-100 Hz)
        idx = f >= 20 & f <= 100;
        powerHF = sum(pxx(idx));
        totalPower = sum(pxx);
        if (powerHF / totalPower) > 0.5 % Threshold can be adjusted
            artifactComponents = [artifactComponents, comp];
        end
    end
    % Remove artifact components
    icasigClean = icasig;
    icasigClean(artifactComponents, :) = 0;
    % Reconstruct the signal
    yClean = (A * icasigClean)';
    
    % 4. Baseline correction (relative to pre-stimulus window)
    trigIndices = find(trig ~= 0);  % Find the indices of non-zero triggers
    preStimulusSamples = round(epochWindow(1) * fs);  % Pre-stimulus in samples
    postStimulusSamples = round(epochWindow(2) * fs); % Post-stimulus in samples
    
    epochs = []; % To store the epoched data
    triggers = []; % To store the corresponding triggers
    
    for j = 1:length(trigIndices)
        startIdx = trigIndices(j) + preStimulusSamples;
        endIdx = trigIndices(j) + postStimulusSamples;
        
        % Make sure we don't exceed the data bounds
        if startIdx > 0 && endIdx <= size(yClean, 1)
            epoch = yClean(startIdx:endIdx, :);  % Extract the epoch
            baseline = mean(yClean(startIdx:trigIndices(j)-1, :), 1);  % Baseline (mean before stimulus)
            epochCorrected = epoch - baseline;  % Baseline correction
            
            epochs = cat(3, epochs, epochCorrected); % Append the epoch
            triggers = [triggers; trig(trigIndices(j))]; % Store the corresponding trigger
        end
    end
    
    % 5. Normalize the data (z-score normalization for each channel)
    epochsNormalized = zscore(epochs, 0, 1);
    
    % Save the pre-processed data to the output folder
    preprocessedFilePath = fullfile(outputFolder, strcat('Preprocessed_', dataFiles(i).name));
    save(preprocessedFilePath, 'epochsNormalized', 'triggers', 'fs', '-v7.3');
    
    fprintf('Processed and saved: %s\n', dataFiles(i).name);
end