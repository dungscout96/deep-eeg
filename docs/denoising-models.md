# Denoising Model Comparisons

The different directories in this respository represent techniques for noise reduction of source data.

## Techniques Overview

### Deep Interpolation

```
Lecoq et al., “Removing Independent Noise in Systems Neuroscience Data Using DeepInterpolation.”
```

- **Architecture**: UNet (2D) specialized to the response times of each modality
- **Demonstrated Data**: fMRI, spiking electrical, spiking optical
- **Notes**:
    - Input structured to probe (e.g. Neuropixel 192 x 2)
    - Standard UNET architecture; N_pre and N_post "designed" to capture minimum feature of interest (e.g. 1ms for spiking)
    - Center frame exclusion: `| -- N_Pre -- || Skip Param || -- Center -- || Skip Param || -- N_Post -- |`
    - Spiking shown to have limited success: "benefits" have caveats where existing spikes are attenuated (some to subthreshold)
    - Primary result is on _image_ techniques (e.g. fMRI, Ca+2 imaging)

### Deep Separator

```
Yu, Junjie, et al. "Embedding Decomposition for Artifacts Removal in EEG Signals." arXiv preprint arXiv:2112.00989 (2021).
```
- **Architecture**: Supervised Ensemble: FCN, LSTM, CNN ("simple" and "complex")
- **Demonstrated Data**: EEG
- **Notes**:
    - Single channel technique
    - Used automated IC Label-based pipeline to generate "clean" EEG, EOG, and EMG signals
    - Trained model might be applicable to novel data (not analyzed)

### EEG Denoise Net

```
Zhang, Haoming, et al. "Eegdenoisenet: A benchmark dataset for deep learning solutions of eeg denoising." Journal of Neural Engineering 18.5 (2021): 056057.
```

- **Architecture**: "Novel" CNN (?)
- **Demonstrated Data**: EEG
- **Notes**:
    - Used automated IC Label-based pipeline to generate "clean" EEG, EOG, and EMG signals (Same as DeepSeparator)
