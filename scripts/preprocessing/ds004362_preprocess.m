function ds004362_preprocess()
addpath('/home/jovyan/deep-eeg/eeglab');
eeglab nogui;
dataset_path = '/expanse/projects/nemar/openneuro/ds004362';
outputDir = './ds004362_imported';

if exist(fullfile(outputDir, 'ds004362.study'))
        [STUDY, ALLEEG] = pop_loadstudy(fullfile(outputDir, 'ds004362.study'));
else
        [STUDY, ALLEEG] = pop_importbids(dataset_path, 'bidsevent','on','bidschanloc', 'on','studyName','ds004362','outputdir', outputDir);
end
outpath = './ds004362_feats';
log_path = './ds004362_logs';
p = gcp('nocreate');
if isempty(p)
   parpool([1 128]);
end
runs = {'run-3' 'run-4' 'run-7' 'run-8' 'run-11' 'run-12'};
parfor i=1:numel(ALLEEG)
    EEG = ALLEEG(i);
    if contains(EEG.filename, runs)
        EEG = pop_loadset('filepath', EEG.filepath, 'filename', EEG.filename);
        ds004362_preprocess_singleset(EEG, outpath, log_path);
    end
end


