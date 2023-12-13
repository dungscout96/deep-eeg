function EEG = ds004362_preprocess_singleset(EEG, outpath, log_path)
    try
        oriChanlocs = EEG.chanlocs;
        options = {'FlatlineCriterion',5,'ChannelCriterion',0.85, ...
                    'LineNoiseCriterion',4,'Highpass',[0.75 1.25], ...
                    'WindowCriterion',0.25,'BurstRejection','off','Distance','Euclidian', ...
                    'WindowCriterionTolerances',[-Inf 7]}; % run channel removal and ASR with default values
        EEG = pop_clean_rawdata(EEG, options{:});      
        EEG = eeg_interp(EEG, oriChanlocs);
        data = [];
        data.data = EEG.data;
        data.evt_markers_names = {EEG.event.type};
        data.evt_markers_sample = [EEG.event.latency];
        data.channames = {EEG.chanlocs.labels};
                
        save(fullfile(outpath, EEG.filename(1:end-4)), '-struct', 'data');        
    catch ME
            errID = fopen(fullfile(log_path, EEG.filename(1:end-4)), 'w');
            fprintf(errID, '%s\n', fullfile(EEG.filepath,EEG.filename));
            fprintf(errID, '%s\n%s\n',ME.identifier, ME.getReport());
            fclose(errID);
    end
end


