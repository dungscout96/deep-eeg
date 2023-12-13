function run_asr_mahnob
    if ~exist('eeglab', 'dir')
        status = system('bash get_eeglab.sh');
        if status ~= 0
            error('Failed to install eeglab and/or necessary plugin')
        end
    end
    addpath('./eeglab');
    eeglab nogui;

    % loop through all sessions
    raw_session_path = '/net2/expData/affective_eeg/mahnob_dataset/Sessions';
    outpath = '/net2/derData/affective_eeg/eeg/raw_asr_mat';

    sessions = dir(raw_session_path);
    save_fun = @save_data;

    err_fid = fopen('run_asr_mahnob.err', 'w');
    fprintf(err_fid,'Failed sessions:\n');
    fclose(err_fid);
    parfor s=1:numel(sessions)
        sesh = sessions(s);

        if ~strcmp(sesh.name, '.') && ~strcmp(sesh.name, '..')
            bdfs = dir(fullfile(raw_session_path, sesh.name, '*.bdf'));
            if length(bdfs) == 1 % each session should have only one bdf file
                try
                    EEG = pop_biosig(fullfile(bdfs.folder, bdfs.name));
                    EEG = pop_eegfiltnew(EEG, 'locutoff', 4, 'hicutoff', 45); % bandpass filter
                    EEG = pop_select(EEG, 'chantype', 'EEG'); % retain only EEG channels
                    EEG = clean_asr(EEG,5, max(0.5, 1.5*EEG.nbchan/EEG.srate), [], 0.66, 'off', [-3.5 5.5], 1, false,false,64); % run ASR with default parameters (made explicitly)

                    data = [];
                    data.data = EEG.data;
                    data.ch_names = {EEG.chanlocs.labels};
                    save_fun(fullfile(outpath,sprintf('%s', sesh.name)), data);
                catch
                    err_fid = fopen('run_asr_mahnob.err', 'a');
                    fprintf(err_fid, '%s\n', sesh.name);
                    fclose(err_fid);
                end
            end
        end
    end

    function save_data(path, data)
        save(path, '-struct', 'data');
    end
end
