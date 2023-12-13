function run_asr_bciiv
    if ~exist('eeglab', 'dir')
        status = system('bash get_eeglab.sh');
        if status ~= 0
            error('Failed to install eeglab and/or necessary plugin')
        end
    end
    addpath('./eeglab');
    eeglab nogui;

    % loop through all sessions
    raw_session_path = '/net2/expData/deep_eeg/bci-iv-1'; % only real subjects data. Synthetic data has been removed
    outpath = '/net2/derData/deep-eeg/bci-iv-1/asr-cleaned';

    sessions = dir(raw_session_path);
    sessions = {'1a', '1b', '1f', '1g'};
    save_fun = @save_data;

    err_fid = fopen('run_asr_bciiv.err', 'w');
    fprintf(err_fid,'Failed sessions:\n');
    fclose(err_fid);
    for s=1:numel(sessions)
        sesh = sessions{s};
        data = load(fullfile(raw_session_path, ['BCICIV_eval_ds' sesh '_1000Hz.mat']));
        
        EEG = eeg_emptyset();
        EEG.data = 0.1*double(data.cnt');
        EEG.filename = ['BCICIV_eval_ds' sesh '_1000Hz.set'];
        EEG.filepath = raw_session_path;
        EEG.srate = 1000;
        EEG.nbchan = numel(data.nfo.clab);
        chanlocs = [];
        for c=1:numel(data.nfo.clab)
            chanloc = [];
            chanloc.labels = data.nfo.clab{c};
            chanloc.X = data.nfo.xpos;
            chanloc.Y = data.nfo.ypos;
            chanlocs = [chanlocs chanloc];
        end
        EEG.chanlocs = chanlocs;
        EEG = eeg_checkset(EEG);
        EEG = pop_resample(EEG, 100);
        EEG = pop_eegfiltnew(EEG, 'locutoff', 4, 'hicutoff', 49); % bandpass filter
        EEG = clean_asr(EEG,5, max(0.5, 1.5*EEG.nbchan/EEG.srate), [], 0.66, 'off', [-3.5 5.5], 1, false,false,64); % run ASR with default parameters (made explicitly)

        data = [];
        data.data = EEG.data;
        data.ch_names = {EEG.chanlocs.labels};
        data.chanX = EEG.chanlocs.X;
        data.chanY = EEG.chanlocs.Y;
        
        marker = load(fullfile(raw_session_path, ['BCICIV_eval_ds' sesh '_1000Hz_true_y.mat']));
        marker = marker.true_y;
        marker = marker(1:10:end);
        size(marker);
        data.marker = marker;
        save(fullfile(outpath, ['BCICIV_eval_ds' sesh '.mat']), '-struct', 'data');
    end
end

