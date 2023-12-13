import os

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

class MahnobTransform(ABC):
    def __init__(self, x_params):
        montage = mne.channels.make_standard_montage('biosemi32')
        self.info = mne.create_info(montage.ch_names, 256, ['eeg',]*len(montage.ch_names))
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

class AvgFreqPower(MahnobTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"]

    def __init__(self, x_params):
        """
        @param dict x_params {
                window: int | None
                stride: int | None
            }
        """
        super().__init__(x_params)
        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

    def transform(self, data):
        normalized_data = {}
        for k in self.data_keys:
            pwr_data = data[k].copy() # prevent shallow changes
            pwr_data_flatten = pwr_data.flatten()
            pwr_data_flatten = scipy.stats.zscore(pwr_data_flatten);
            normalized_data[k] = np.reshape(pwr_data_flatten, (pwr_data.shape[0],pwr_data.shape[1]))

        # Assumes all data_keys have same dimensionality
        shape_time = normalized_data[self.data_keys[0]].shape[0]

        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.mean(
                np.take(normalized_data[k], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False) # SLICE F
            for k in self.data_keys] # KEYS SLICE F
        
        tensor = np.array(d)
        tensor = np.transpose(tensor, (1,0,2)) # SLICE KEYS F
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2])) # SLICE F
        return tensor


class AvgFreqPowerHighDensity(MahnobTransform):
    data_keys = ["feat_inst_narrow"]

    def __init__(self, x_params):
        """
        @param dict x_params {
                window: int | None
                stride: int | None
            }
        """
        super().__init__(x_params)
        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

    def transform(self, data):
        normalized_data = []
        for d in data[self.data_keys[0]]:
            pwr_data = d.copy() # prevent shallow changes
            pwr_data_flatten = pwr_data.flatten()
            pwr_data_flatten = scipy.stats.zscore(pwr_data_flatten);
            normalized_data.append(np.reshape(pwr_data_flatten, (pwr_data.shape[0],pwr_data.shape[1])))
        normalized_data = np.array(normalized_data)

        shape_time = normalized_data.shape[1]
        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.mean(
                np.take(normalized_data[i], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False) # SLICE F
            for i in range(len(normalized_data))] # NARROW_BINS SLICE F
        
        tensor = np.array(d)
        tensor = np.transpose(tensor, (1,0,2)) # SLICE NARROW_BINS F
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2])) # SLICE F
        return tensor


class TopomapNN(MahnobTransform, NNUtils):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    
    def __init__(self, x_params):
        """
        @param dict x_params {
                model: str
                model_params: dict (arguments passed directly to model constructor)
                model_augment_fn: fn | None
                window: int | None
                stride: int | None
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
        
        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

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
                np.ones(32), self.info, ch_type='eeg', sensors=False, contours=0,
                cmap="Reds", outlines='head', size=4, show=False)
            fig_mask = plt.gcf()
            fig_mask.canvas.draw()
            plt.close()
            fig_mask_data = np.array(np.asarray(fig_mask.canvas.buffer_rgba())[:,:,0])
            mask = np.array((fig_mask_data == 103), dtype=np.uint8)
            buffer = buffer * np.expand_dims(mask, axis=-1)
        
        return buffer[50:-50,50:-50,:] # trim padding and return

    def __generate_network_feat(self, data):
        # Assumes all data_keys have same dimensionality
        shape_time = data[self.data_keys[0]].shape[0]

        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.mean(
                np.take(data[k], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False) # SLICE F
            for k in self.data_keys] # KEYS SLICE F

        d = np.transpose(d, (1,0,2)) # SLICE KEYS F
        tensor = [] # SLICE F..
        for i in range(d.shape[0]):
            tensor.append(self.__topomap_tensor_mne(d[i], mask=True)) # F..
        
        return np.array(tensor)

    def transform(self, data):
        feat_tensor = torch.tensor(self.__generate_network_feat(data),
            dtype=torch.float,
            device="cuda" if torch.cuda.is_available() else "cpu") # T F
        feat_tensor = feat_tensor.transpose(1,3)
        feat_tensor = torch.nn.functional.interpolate(feat_tensor, size=(224,224))
        with torch.no_grad():
            result = self.model(feat_tensor)
        del feat_tensor # free gpu memory

        result = result.numpy(force=True) # always force back to cpu and np
        return result # T F


class TopomapImg(MahnobTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    
    def __init__(self, x_params):
        """
        @param dict x_params {
                model_params: dict | { n_projections: int }
                window: int | None
                stride: int | None
            }
        """
        super().__init__(x_params)
        
        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

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
                np.ones(32), self.info, ch_type='eeg', sensors=False, contours=0,
                cmap="Reds", outlines='head', size=4, show=False)
            fig_mask = plt.gcf()
            fig_mask.canvas.draw()
            plt.close()
            fig_mask_data = np.array(np.asarray(fig_mask.canvas.buffer_rgba())[:,:,0])
            mask = np.array((fig_mask_data == 103), dtype=np.uint8)
            buffer = buffer * np.expand_dims(mask, axis=-1)
        
        return buffer[50:-50,50:-50,:] # trim padding and return

    def __generate_network_feat(self, data):
        # Assumes all data_keys have same dimensionality
        shape_time = data[self.data_keys[0]].shape[0]

        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.mean(
                np.take(data[k], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False) # SLICE F
            for k in self.data_keys] # KEYS SLICE F

        d = np.transpose(d, (1,0,2)) # SLICE KEYS F
        tensor = [] # SLICE F..
        for i in range(d.shape[0]):
            tensor.append(self.__topomap_tensor_mne(d[i], mask=True)) # F..
        
        return np.array(tensor)

    def transform(self, data):
        return self.__generate_network_feat(data) # T F

class Spectrogram(MahnobTransform):
    data_keys = ["data"]

    def __init__(self, x_params):
        """
        @param dict x_params {
                window: int | None
                stride: int | None
            }
        """
        super().__init__(x_params)
        self.model = getattr(torchmodels, x_params["model"])(**x_params["model_params"])

        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

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
            MahnobDataset.SFREQ, freqs, 
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
        return result

    def transform(self, data):
        normalized_data = []
        for k in self.data_keys:
            normalized_data.append(scipy.stats.zscore(data[k], axis=None))

        # Assumes all data_keys have same dimensionality
        shape_time = normalized_data[0].shape[0]

        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.take(normalized_data[i], range_idxs, axis=0) # SLICE WINDOW F
                for i in range(len(normalized_data))] # K SLICE WINDOW F
        
        tensor = np.array(d)
        tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])) # SLICE WINDOW F
        tfs = self._compute_tf(tensor) # SLICE CH FREQS Time
        return self.__generate_network_feat(tfs)

    
class SpectrogramImg(MahnobTransform):
    data_keys = ["data"]

    def __init__(self, x_params):
        """
        @param dict x_params {
                window: int | None
                stride: int | None
            }
        """
        super().__init__(x_params)
        self.window = None if x_params["window"] < 1 else x_params["window"]
        self.stride =  None if self.window is None else x_params["stride"]

    def _compute_tf(self, data):
        if len(data.shape) > 3:
            raise ValueError('Only accept slice x time x ch data')
        if data.shape[2] > data.shape[1]:
            raise ValueError('Data is not time x chan')

        freqs = np.array(range(1,50))
        tfs = mne.time_frequency.tfr_array_morlet(
            np.transpose(data, (0,2,1)), 
            MahnobDataset.SFREQ, freqs, 
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

            # resize to match VGG16 expected input dim
            channel_batch_feat_tensor = torch.tensor(np.array(channel_batch),
                                       dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu")
            channel_batch_feat_tensor = torch.nn.functional.interpolate(channel_batch_feat_tensor, size=(224,224))

            result = channel_batch_feat_tensor.numpy(force=True)
            del channel_batch_feat_tensor # clear GPU
            batch.append(result)
            
        return np.array(batch)

    def transform(self, data):
        normalized_data = []
        for k in self.data_keys:
            normalized_data.append(scipy.stats.zscore(data[k], axis=None))

        # Assumes all data_keys have same dimensionality
        shape_time = normalized_data[0].shape[0]

        effective_window = shape_time if self.window is None else self.window
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.take(normalized_data[i], range_idxs, axis=0) # SLICE WINDOW F
                for i in range(len(normalized_data))] # K SLICE WINDOW F
        
        tensor = np.array(d)
        tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])) # SLICE WINDOW F
        tfs = self._compute_tf(tensor) # SLICE CH FREQS Time
        return self.__generate_network_feat(tfs)

    
class MahnobDataset(torch.utils.data.Dataset):
    SFREQ = 256
    BEHAVIOR_KEYS = ("feltVlnc", "feltArsl", "feltCtrl", "feltPred")

    def __init__(self, 
            data_dir='/net2/derData/affective_eeg/align/asr_eeg_beh', # location of asr cleaned data bundled with markers and channels
            sessions:list=None,                                       # sessions to use, default to all
            n_sessions=None,                                          # number of sessions to pick, default all
            y_mode="bimodal", # {ordinal, split, bimodal}             # y to return (filtered on only first y_key) - ordinal: full range, split: <5 & >5, bimodal: <=3, >=7
            y_keys=["feltVlnc"],                                      # feltVlnc, feltArsl, feltCtrl, feltPred
            x_params={
                "feature": "TopomapVggPretrained",                    # 
                "window": -1,                                         # number of samples to average over (-1: full session)
                "stride": 1,                                          # number of samples to stride window by (does nothing when window = -1)
            },
            balanced=False,                                           # within k-cv only; enforce class balance y_mode (only for split and bimodal); only on first y_key
            n_cv=None,                                                # (k,folds) Use this to control train vs test; independent of seed
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir
        self.sessions = [i.split('.')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'mat']
        if sessions != None:
            self.sessions = [i for i in sessions if i in self.sessions]
            if len(sessions) - len(self.sessions) > 0:
                print("Warning: unknown keys present in user specified sessions")
        np.random.shuffle(self.sessions)
        n_sessions = n_sessions if n_sessions is not None else len(self.sessions)
        if n_sessions > len(self.sessions):
            print("Warning: n_sessions cannot be larger than sessions")
        self.sessions = self.sessions[:n_sessions]

        # Split Train-Test
        self.n_cv = n_cv
        self.is_test = is_test
        if self.n_cv is not None:
            split_size = int(n_sessions / self.n_cv[1])
            if not self.is_test:
                self.sessions = self.sessions[:self.n_cv[0]*split_size] + \
                                self.sessions[(self.n_cv[0]+1)*split_size:]
            else:
                self.sessions = self.sessions[self.n_cv[0]*split_size:(self.n_cv[0]+1)*split_size]

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
            if y_key not in MahnobDataset.BEHAVIOR_KEYS:
                raise ValueError(f"Invalid y_keys: {y_key}")
        self.y_keys = y_keys

        self.balanced = balanced
        if self.balanced and self.y_mode not in ("split", "bimodal"):
            print(f"WARNING: ignoring balanced flag for y_mode={self.y_mode}")

        # Preload data
        raw_data = [self.__preload_raw(session) for session in tqdm(self.sessions)]
        # Transform data (populates self.data and self.ch_names). Note: this modifies input data.
        self.__transform_raw(raw_data)
        # Store y-data (all behavioral data is noted consistent across time) and modidy data as appropriate
        self.__transform_y(raw_data)

        if x_params["feature"] == "SpectrogramImg":
            # treat channel as sample for spectrogram images feature
            if len(self.data.shape) == 6 and self.data.shape[1] == 1:
                self.data = np.squeeze(self.data, 1)
            if len(self.data.shape) == 5 and self.y_data.shape[0] == self.data.shape[0]:
                self.y_data = np.repeat(self.y_data, self.data.shape[1], axis=1)  # expand T dim of label
                self.y_data = np.reshape(self.y_data,(-1,1)) # collapse T into batch size
                
            if len(self.sessions.shape) == 1:
                self.sessions = np.expand_dims(self.sessions, axis=1)
                self.sessions = np.repeat(self.sessions, self.data.shape[1], axis=1)  # expand channel for session
                self.sessions = np.reshape(self.sessions,(-1,1)) # collapse T into batch size
                
            if len(self.data.shape) == 5:
                self.data = np.reshape(self.data, (self.data.shape[0]*self.data.shape[1],*self.data.shape[2:])) # collapse time dimension
            print('session', self.sessions.shape, 'data', self.data.shape, 'label', self.y_data.shape)

    def __preload_raw(self, session):
        # print(f"Preloading {session}...")
        mat_data = scipy.io.loadmat(os.path.join(self.basedir, session))
        # only keep relevant keys
        mat_data = {k: v for k,v in mat_data.items() if k in self.y_keys or k == "ch_names" or k in self.x_transformer.data_keys}
        # check to ensure time major
        for k in self.x_transformer.data_keys:
            if mat_data[k].shape[-2] > mat_data[k].shape[-1]:
                print("WARNING: data looks to already be Time Major")
            else: 
                mat_data[k] = np.swapaxes(mat_data[k], -1, -2)
        return mat_data

    def __transform_raw(self, data):
        # Transform channels to be consistent across sessions
        self.ch_names = self.x_transformer.info['ch_names']
        for n_session in range(len(data)):
            session_ch_names = [ch.strip() for ch in data[n_session]['ch_names']] # matlab forces fixed len str
            ch_order = [session_ch_names.index(i) for i in self.ch_names]
            for k in self.x_transformer.data_keys:
                data[n_session][k] = data[n_session][k][:, ch_order] # T Ch

        # Transform to feature space
        self.data = [self.x_transformer.transform(d) for d in tqdm(data)] # S T F..
        
        # If T is unequal, reconcile
        time_shapes = np.array([np.shape(d)[0] for d in self.data])
        if not np.all(time_shapes == time_shapes[0]):
            n_windows = min(time_shapes)
            print(f"INFO: T dimension is not equivalent across all S; reducing to T={n_windows}")
            self.data = [d[-n_windows:] for d in self.data]
        self.data = np.array(self.data)

    def __transform_y(self, data):
        self.y_data = [[i[k][0] for k in self.y_keys] for i in data]
        self.y_data = np.array(self.y_data, dtype=np.int8)
        self.y_data = np.squeeze(self.y_data, axis=-1) # squash last dim (S len(y_keys))

        if self.y_mode == "ordinal":
            return
        elif self.y_mode == "split":
            # filtered only on first y_key
            keep_idxs = [n for n,y in enumerate(self.y_data) if y[0] != 5]
            self.y_data = self.y_data[keep_idxs]
            self.y_data = np.array([[int(i > 5)  for i in y] for y in self.y_data])
            self.data = self.data[keep_idxs]
            self.sessions = np.array([self.sessions[i] for i in keep_idxs])
        elif self.y_mode == "bimodal":
            keep_idxs = [n for n,y in enumerate(self.y_data) if y[0] <= 3 or y[0] >= 7]
            self.y_data = self.y_data[keep_idxs]
            self.y_data = np.array([[int(i > 5)  for i in y] for y in self.y_data])
            self.data = self.data[keep_idxs]
            self.sessions = np.array([self.sessions[i] for i in keep_idxs])

        # Balance on first y_key
        if self.balanced:
            y_key_0 = [y[0] for y in self.y_data]
            counter = [0, 0]
            y_counts = []
            for i in y_key_0:
                y_counts.append(counter[i])
                counter[i] += 1
            keep_idxs = np.argwhere(np.array(y_counts) < min(y_key_0.count(0), y_key_0.count(1))).squeeze()
            self.y_data = self.y_data[keep_idxs]
            self.data = self.data[keep_idxs]
            self.sessions = np.array([self.sessions[i] for i in keep_idxs])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.y_data[idx]
