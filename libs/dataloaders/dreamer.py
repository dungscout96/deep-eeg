import os
import pickle
import mne
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision.models as torchmodels
from tqdm import tqdm
from matplotlib import pyplot as plt

from abc import ABC, abstractmethod
from .utils import NNUtils
import matplotlib.cm as cm

class DreamerTransform(ABC):
    data_key_map = {
        "feat_inst_theta": (4., 8.),
        "feat_inst_alpha": (8., 13.),
        "feat_inst_beta": (13., 20.),
    }

    def __init__(self, x_params):
        # This is the Emotiv ordering
        ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        montage = mne.channels.make_standard_montage('standard_1020') # create default montage
        info = mne.create_info(ch_names, DreamerDataset.SFREQ, ['eeg',]*len(ch_names)) # wrap montage in info
        info.set_montage(montage) # strip unused channels from default
        montage = info.get_montage() # extract montage

        # Could just reuse the info object, but letting mne do any channel name cleanup
        self.info = mne.create_info(montage.ch_names, DreamerDataset.SFREQ, ['eeg',]*len(montage.ch_names))
        mne.channels.montage._set_montage(self.info, montage)

    @property
    @abstractmethod
    def data_keys(self):
        pass

    @abstractmethod
    def transform(self, data):
        """
        @param dict data
        @return np.array - time x features
        """
        pass

class AvgFreqPower(DreamerTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"]


    def __init__(self, x_params):
        """
        @param dict x_params {
            }
        """
        super().__init__(x_params)

    def transform(self, data):
        normalized_data = {}
        for k in self.data_keys:
            filt_data = [mne.filter.filter_data(d.T, sfreq=DreamerDataset.SFREQ, 
                            l_freq=DreamerTransform.data_key_map[k][0], 
                            h_freq=DreamerTransform.data_key_map[k][1], 
                            l_trans_bandwidth=1., h_trans_bandwidth=1.,
                            n_jobs=16, method='fir', phase='zero', copy=True, verbose=False)
                        for d in data] # S F T
            pwr_data = np.array([np.abs(scipy.signal.hilbert(d))**2 for d in filt_data])

            pwr_data_flatten = pwr_data.flatten()
            pwr_data_flatten = scipy.stats.zscore(pwr_data_flatten);
            normalized_data[k] = np.reshape(pwr_data_flatten, pwr_data.shape) # S F T

        d = [np.mean(normalized_data[k], axis=2, keepdims=False) for k in self.data_keys] # KEYS S F
        tensor = np.array(d)
        tensor = np.transpose(tensor, (1,0,2)) # S KEYS F
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2])) # S F
        tensor = np.expand_dims(tensor, axis=1) # S T(1) F
        return tensor


class TopomapNN(DreamerTransform, NNUtils):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    
    def __init__(self, x_params):
        """
        @param dict x_params {
                model: str
                model_params: dict (arguments passed directly to model constructor)
                model_augment_fn: fn | None
                random_weights: {
                    "mode": str (zero | rand_init | perturb | shuffle)
                    "distribution": (mode==perturb: gaussian | uniform)
                    "seed": int (optional)
                    }
                }
            }
        """
        super().__init__(x_params)
        self.model = getattr(torchmodels, x_params["model"])(**x_params["model_params"])

        if "random_weights" in x_params:
            self.randomize_weights(x_params["random_weights"])

        if "model_augment_fn" in x_params and x_params["model_augment_fn"] is not None:
            x_params["model_augment_fn"](self.model)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def __topomap_tensor_mne(self, data_rgb, mask):
        """
        Returns a 300x300x3 RGB topo data array projected onto a head model
        """
        figs = []
        
        for data, color in zip(data_rgb, ["Reds", "Greens", "Blues"]):
            mne.viz.plot_topomap(
                data, self.info, ch_type='eeg', sensors=False, contours=0, 
                cmap=color, outlines='head', size=4, show=False)
            fig = plt.gcf()
            fig.canvas.draw()
            figs.append(fig)
            plt.close()
        
        # Convert to RGB array
        fig_data = np.array([np.asarray(fig.canvas.buffer_rgba())[:,:,n] for n,fig in enumerate(figs)])
        buffer = np.transpose(fig_data, axes=(1,2,0))
        
        if mask:
            mne.viz.plot_topomap(
                np.ones(14), self.info, ch_type='eeg', sensors=False, contours=0,
                cmap="Reds", outlines='head', size=4, show=False)
            fig_mask = plt.gcf()
            fig_mask.canvas.draw()
            plt.close()
            fig_mask_data = np.array(np.asarray(fig_mask.canvas.buffer_rgba())[:,:,0])
            mask = np.array((fig_mask_data == 103), dtype=np.uint8)
            buffer = buffer * np.expand_dims(mask, axis=-1)
        
        return buffer[50:-50,50:-50,:] # trim padding and return

    def __generate_network_feat(self, data):
        normalized_data = {}
        for k in self.data_keys:
            filt_data = [mne.filter.filter_data(d.T, sfreq=DreamerDataset.SFREQ, 
                            l_freq=DreamerTransform.data_key_map[k][0], 
                            h_freq=DreamerTransform.data_key_map[k][1], 
                            l_trans_bandwidth=1., h_trans_bandwidth=1.,
                            n_jobs=16, method='fir', phase='zero', copy=True, verbose=False)
                        for d in data] # S F T
            pwr_data = np.array([np.abs(scipy.signal.hilbert(d))**2 for d in filt_data])

            pwr_data_flatten = pwr_data.flatten()
            pwr_data_flatten = scipy.stats.zscore(pwr_data_flatten);
            normalized_data[k] = np.reshape(pwr_data_flatten, pwr_data.shape) # S F T

        d = [np.mean(normalized_data[k], axis=2, keepdims=False) for k in self.data_keys] # KEYS S F
        d = np.array(d)
        d = np.transpose(d, (1,0,2)) # S KEYS F
        
        tensor = [] # S F..
        for i in range(d.shape[0]):
            tensor.append(self.__topomap_tensor_mne(d[i], mask=True)) # F..
        tensor = np.array(tensor)
        return tensor

    def transform(self, data):
        feat_tensor = torch.tensor(self.__generate_network_feat(data),
            dtype=torch.float,
            device="cuda" if torch.cuda.is_available() else "cpu") # S F
        feat_tensor = feat_tensor.transpose(1,3)
        feat_tensor = torch.nn.functional.interpolate(feat_tensor, size=(224,224))
        with torch.no_grad():
            result = self.model(feat_tensor)
        del feat_tensor # free gpu memory

        result = result.numpy(force=True) # always force back to cpu and np
        tensor = np.expand_dims(result, axis=1) # S T(1) F..
        return tensor # S T(1) F


class TopomapImg(DreamerTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    
    def __init__(self, x_params):
        """
        @param dict x_params {
                model_params: dict | { n_projections: int }
            }
        """
        super().__init__(x_params)

    def __topomap_tensor_mne(self, data_rgb, mask):
        """
        Returns a 300x300x3 RGB topo data array projected onto a head model
        """
        figs = []
        
        for data, color in zip(data_rgb, ["Reds", "Greens", "Blues"]):
            mne.viz.plot_topomap(
                data, self.info, ch_type='eeg', sensors=False, contours=0, 
                cmap=color, outlines='head', size=4, show=False)
            fig = plt.gcf()
            fig.canvas.draw()
            figs.append(fig)
            plt.close()
        
        # Convert to RGB array
        fig_data = np.array([np.asarray(fig.canvas.buffer_rgba())[:,:,n] for n,fig in enumerate(figs)])
        buffer = np.transpose(fig_data, axes=(1,2,0))
        
        if mask:
            mne.viz.plot_topomap(
                np.ones(14), self.info, ch_type='eeg', sensors=False, contours=0,
                cmap="Reds", outlines='head', size=4, show=False)
            fig_mask = plt.gcf()
            fig_mask.canvas.draw()
            plt.close()
            fig_mask_data = np.array(np.asarray(fig_mask.canvas.buffer_rgba())[:,:,0])
            mask = np.array((fig_mask_data == 103), dtype=np.uint8)
            buffer = buffer * np.expand_dims(mask, axis=-1)
        
        return buffer[50:-50,50:-50,:] # trim padding and return

    def __generate_network_feat(self, data):
        normalized_data = {}
        for k in self.data_keys:
            filt_data = [mne.filter.filter_data(d.T, sfreq=DreamerDataset.SFREQ, 
                            l_freq=DreamerTransform.data_key_map[k][0], 
                            h_freq=DreamerTransform.data_key_map[k][1], 
                            l_trans_bandwidth=1., h_trans_bandwidth=1.,
                            n_jobs=16, method='fir', phase='zero', copy=True, verbose=False)
                        for d in data] # S F T
            pwr_data = np.array([np.abs(scipy.signal.hilbert(d))**2 for d in filt_data])

            pwr_data_flatten = pwr_data.flatten()
            pwr_data_flatten = scipy.stats.zscore(pwr_data_flatten);
            normalized_data[k] = np.reshape(pwr_data_flatten, pwr_data.shape) # S F T

        d = [np.mean(normalized_data[k], axis=2, keepdims=False) for k in self.data_keys] # KEYS S F
        d = np.array(d)
        d = np.transpose(d, (1,0,2)) # S KEYS F
        
        tensor = [] # S F..
        for i in range(d.shape[0]):
            tensor.append(self.__topomap_tensor_mne(d[i], mask=True)) # F..
        tensor = np.array(tensor)
        return tensor

    def transform(self, data):
        result = self.__generate_network_feat(data) # S F
        tensor = np.expand_dims(result, axis=1) # S T(1) F..
        return tensor # S T(1) F

class Spectrogram(DreamerTransform):
    data_keys = ["data"]

    def __init__(self, x_params):
        """
        @param dict x_params { ... }
        """
        super().__init__(x_params)
        self.model = getattr(torchmodels, x_params["model"])(**x_params["model_params"])

        if "random_weights" in x_params:
            self.randomize_weights(x_params["random_weights"])

        if "model_augment_fn" in x_params and x_params["model_augment_fn"] is not None:
            x_params["model_augment_fn"](self.model)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def _compute_tf(self, data):
        if len(data.shape) > 3:
            raise ValueError('Only accept slice x time x ch data')
        if data.shape[2] > data.shape[1]:
            raise ValueError('Data is not time x chan')

        freqs = np.array(range(1,50))
        tfs = mne.time_frequency.tfr_array_morlet(
            np.transpose(data, (0,2,1)), 
            DreamerDataset.SFREQ, freqs, 
            n_cycles=freqs/2, 
            output="power",
            n_jobs=4)
        return tfs

    def __generate_network_feat(self, data):
        # @param data - SLICE CH FREQS Time
        batch = []
        for i in range(data.shape[0]):
            for c in range(data.shape[1]):
                tf = data[i,c,:,:] # channel psd
                # convert psd value to rgb
                sm = cm.ScalarMappable(cmap='jet')
                sm.set_clim(tf.min(), tf.max())
                im = sm.to_rgba(tf)[:,:,:3] # truncating alpha
                im = np.transpose(im, (2,0,1)) # 3 F T
                batch.append(im)

        # resize to match VGG16 expected input dim
        batch_feat_tensor = torch.tensor(np.array(batch),
                                   dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu")
        batch_feat_tensor = torch.nn.functional.interpolate(batch_feat_tensor, size=(224,224))

        with torch.no_grad():
            chs_embed = self.model(batch_feat_tensor) # SLICE*CH F
        result = torch.reshape(chs_embed, (data.shape[0], -1))
        result = result.numpy(force=True)
        return np.expand_dims(result, axis=1)

    def transform(self, data):
        normalized_data = scipy.stats.zscore(data, axis=None) # S T F
        tfs = self._compute_tf(normalized_data) # SLICE CH FREQS Time
        return self.__generate_network_feat(tfs)

class SpectrogramImg(DreamerTransform):
    data_keys = ["data"]

    def __init__(self, x_params):
        """
        @param dict x_params { ... }
        """
        super().__init__(x_params)

    def _compute_tf(self, data):
        if len(data.shape) > 3:
            raise ValueError('Only accept slice x time x ch data')
        if data.shape[2] > data.shape[1]:
            raise ValueError('Data is not time x chan')

        freqs = np.array(range(1,50))
        tfs = mne.time_frequency.tfr_array_morlet(
            np.transpose(data, (0,2,1)), 
            DreamerDataset.SFREQ, freqs, 
            n_cycles=freqs/2, 
            output="power",
            n_jobs=4)
        return tfs

    def __generate_network_feat(self, data):
        # @param data - SLICE CH FREQS Time
        batch = []
        for i in range(data.shape[0]):
            channel_batch = []
            for c in range(data.shape[1]):
                tf = data[i,c,:,:] # channel psd
                # convert psd value to rgb
                sm = cm.ScalarMappable(cmap='jet')
                sm.set_clim(tf.min(), tf.max())
                im = sm.to_rgba(tf)[:,:,:3] # truncating alpha
                im = np.transpose(im, (2,0,1)) # 3 F T
                channel_batch.append(im)
                
            channel_batch_feat_tensor = torch.tensor(np.array(channel_batch),
                               dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu")
            channel_batch_feat_tensor = torch.nn.functional.interpolate(channel_batch_feat_tensor, size=(224,224))
            result = channel_batch_feat_tensor.numpy(force=True)
            del channel_batch_feat_tensor # clear GPU
            batch.append(result)

        return np.array(batch)

    def transform(self, data):
        normalized_data = scipy.stats.zscore(data, axis=None) # S T F
        tfs = self._compute_tf(normalized_data) # SLICE CH FREQS Time
        return self.__generate_network_feat(tfs)
    
class DreamerDataset(torch.utils.data.Dataset):
    SFREQ = 128
    BEHAVIOR_KEYS = ("valence", "arousal", "dominance", "liking")

    def __init__(self, 
            data_file='/net2/derData/deep-eeg/dreamer/dreamer.mat',
            sessions=None,                                            # Session indices; subject names removed during import
            y_mode="bimodal", # {ordinal, split, bimodal}             # y to return (filtered on only first y_key) - ordinal: full range, split: <5 & >5, bimodal: <=3, >=7
            y_keys=["valence"],                                       # valence, arousal, dominance, liking
            x_params={
                "feature": "TopomapVggPretrained",                    # 
            },
            balanced=False,                                           # enforce class balance y_mode (only for split and bimodal); only on first y_key
            seed=None):                                               # numpy random seed
        
        np.random.seed(seed)

        # Data gets preloaded here due to data bundling format
        raw_mat = scipy.io.loadmat(data_file)
        raw = raw_mat["DREAMER"][0][0]
        # sfreq = float(raw[1].squeeze()) # we hardcode 128Hz elsewhere
        self.sessions = list(np.arange(len(raw[0].squeeze())))
        # baseline_data = [subj[0].squeeze()['EEG'].item().squeeze().item()[0].squeeze() for subj in raw[0].squeeze()] # Subj x S x T x F
        raw_x = [subj[0].squeeze()['EEG'].item().squeeze().item()[1].squeeze() for subj in raw[0].squeeze()] # Subj x S x T x F
        # Per DREAMER data collection paper, they only use the trailing 60 seconds for analysis
        raw_x = np.array([[s[-int(DreamerDataset.SFREQ*60):,:] for s in subj] for subj in raw_x])
        # y data extraction
        y_valence = np.array([list(subj[0].squeeze()['ScoreValence'].item().squeeze()) for subj in raw[0].squeeze()])
        y_arousal = np.array([list(subj[0].squeeze()['ScoreArousal'].item().squeeze()) for subj in raw[0].squeeze()])
        y_dominance = np.array([list(subj[0].squeeze()['ScoreDominance'].item().squeeze()) for subj in raw[0].squeeze()])
        raw_y = np.array([y_valence, y_arousal, y_dominance]) # 3 x subject x S
        raw_y = np.transpose(raw_y, (1,2,0)) # subject x S x 3

        if sessions != None:
            self.sessions = [i for i in sessions if i in self.sessions]
            if len(sessions) - len(self.sessions) > 0:
                print("Warning: unknown keys present in user specified sessions")
        np.random.shuffle(self.sessions)

        # Channel reordering does not need to be done as a custom montage object is created
        # with the ordering specified to be what the emotiv ordering is
        # original_order = [i[0] for i in raw[3].squeeze()] # original channel ordering

        # Instantiate transformer
        self.x_params = x_params
        if type(x_params["feature"]) is str:
            try:
                transformer_cls = globals()[x_params["feature"]]
            except KeyError:
                raise ValueError("x_params.feature class not found")
        else:
            transformer_cls = x_params["feature"]
        self.x_transformer = transformer_cls(self.x_params)

        if y_mode.lower() not in ("ordinal", "split", "bimodal"):
            raise ValueError("Invalid y_mode")
        self.y_mode = y_mode

        for y_key in y_keys:
            if y_key not in DreamerDataset.BEHAVIOR_KEYS:
                raise ValueError(f"Invalid y_keys: {y_key}")
        self.y_keys = y_keys

        self.balanced = balanced
        if self.balanced and self.y_mode not in ("split", "bimodal"):
            print(f"WARNING: ignoring balanced flag for y_mode={self.y_mode}")

        self.__transform_raw(raw_x)
        self.__transform_y(raw_y)
        
        if x_params["feature"] == "SpectrogramImg":
            # treat channel as sample for spectrogram images feature
            if len(self.data.shape) == 6 and self.data.shape[1] == 1:
                self.data = np.squeeze(self.data, 1)
            if len(self.data.shape) == 5 and self.y_data.shape[0] == self.data.shape[0]:
                self.y_data = np.repeat(self.y_data, self.data.shape[1], axis=1)  # expand channel for label
                self.y_data = np.reshape(self.y_data,(-1,1)) # collapse T into batch size
                
                if len(self.sessions.shape) == 1:
                    self.sessions = np.expand_dims(self.sessions, axis=1)
                self.sessions = np.repeat(self.sessions, self.data.shape[1], axis=1)  # expand channel for session
                self.sessions = np.reshape(self.sessions,(-1,1)) # collapse T into batch size
                
            if len(self.data.shape) == 5:
                self.data = np.reshape(self.data, (self.data.shape[0]*self.data.shape[1],*self.data.shape[2:])) # collapse time dimension
            # print('len session', self.sessions.shape, 'len data', self.data.shape, 'len label', self.y_data.shape)
            # print(self.sessions)
            
    def __transform_raw(self, data):

        self.ch_names = self.x_transformer.info['ch_names']
        # Transform to feature space
        self.data = np.array([self.x_transformer.transform(d) for d in tqdm(data)]) # Session S T F..
        flat_sessions = np.array([[s,]*i.shape[0] for i,s in zip(self.data, self.sessions)])
        self.sessions = flat_sessions.flatten()
        shape = self.data.shape
        self.data = np.reshape(self.data, (shape[0]*shape[1], *shape[2:]))

    def __transform_y(self, data):
        y_take = [DreamerDataset.BEHAVIOR_KEYS.index(k) for k in self.y_keys]
        self.y_data = data[:,:, y_take] # subject x S x Take
        self.y_data = np.array(self.y_data, dtype=np.int8)
        shape = self.y_data.shape
        self.y_data = np.reshape(self.y_data, (shape[0]*shape[1], shape[2])) # S T

        if self.y_mode == "ordinal":
            return
        elif self.y_mode == "split":
            self.y_data = np.array([[int(i > 3)  for i in y] for y in self.y_data])
        elif self.y_mode == "bimodal":
            keep_idxs = [n for n,y in enumerate(self.y_data) if y[0] != 3]
            self.y_data = self.y_data[keep_idxs]
            self.y_data = np.array([[int(i > 3)  for i in y] for y in self.y_data])
            self.data = self.data[keep_idxs]
            self.sessions = [self.sessions[i] for i in keep_idxs]

        # Balance on first y_key by session
        if self.balanced:
            y_key_0 = np.array([y[0] for y in self.y_data])
            kidxs = []
            for session in np.unique(self.sessions):
                indices = np.array([n for n,i in enumerate(self.sessions) if session == i])
                y_key_session_0 = list(y_key_0[indices])
                counter = [0, 0]
                y_counts = []
                for i in y_key_session_0:
                    y_counts.append(counter[i])
                    counter[i] += 1
                keep_idxs = np.argwhere(np.array(y_counts) < min(y_key_session_0.count(0), y_key_session_0.count(1))).squeeze()
                print(f"Session {session} count: {len(keep_idxs)}")
                kidxs.extend(indices[keep_idxs])
            self.y_data = self.y_data[kidxs]
            self.data = self.data[kidxs]
            self.sessions = [self.sessions[i] for i in kidxs]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.y_data[idx]