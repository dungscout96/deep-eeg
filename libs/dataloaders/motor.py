import os
import copy
import mne
from scipy.interpolate import interp1d
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision.models as torchmodels
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
from abc import ABC, abstractmethod

from .utils import NNUtils

class MotorTransform(ABC):
    def __init__(self, x_params):
        self.SFREQ = x_params['sfreq']
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        montage = mne.channels.read_custom_montage(os.path.join(__location__, 'electrocap_international_64.locs'), coord_frame="head")
        self.info = mne.create_info(montage.ch_names, self.SFREQ, ['eeg',]*len(montage.ch_names))
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

    def __get_power(self, data, zscore=False):
        """
        Return analytical signal power using hilbert and (optionally) zscored data
        @param data - ch x time
        @param bool zscore - normalize the data per channel across time
        @return np.array power - instantaneous power of same shape as data
        """
        # Normalize each channel (data[i,:]~N(0,1))
        #   (x-mu)/stdev
        if zscore:
            m = np.mean(data, axis=1)
            m = np.tile(m,(data.shape[1],1)).T
            s = np.std(data, axis=1, ddof=1)
            s = np.tile(s, (data.shape[1],1)).T
            data = (data-m)/s
        # Hilbert Transform
        # Note this is the analytical signal, not the hilbert transform
        # To get the hilbert transformed signal: np.imag(hilbert(x))
        #                       original signal: np.real(hilbert(x))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        analytical_signal = scipy.signal.hilbert(data)
        a_power = np.abs(analytical_signal)**2
        # a_phase = np.angle(analytical_signal)
        a_power = data
        return a_power

    def get_spectral_power(self, data):
        feats = []
        for n_session in tqdm(range(len(data))):
            raw = np.array(data[n_session]['data']).astype(float)
            theta = mne.filter.filter_data(raw, sfreq=float(self.SFREQ), l_freq=4., h_freq=8., l_trans_bandwidth=1., h_trans_bandwidth=1.,
                            n_jobs=4, method='fir', phase='zero', copy=True, verbose=False)
            alpha = mne.filter.filter_data(raw, sfreq=float(self.SFREQ), l_freq=8., h_freq=13., l_trans_bandwidth=1., h_trans_bandwidth=1.,
                                n_jobs=4, method='fir', phase='zero', copy=True, verbose=False)
            beta = mne.filter.filter_data(raw, sfreq=float(self.SFREQ), l_freq=10., h_freq=35., l_trans_bandwidth=1., h_trans_bandwidth=1.,
                                n_jobs=4, method='fir', phase='zero', copy=True, verbose=False)
            gamma = mne.filter.filter_data(raw, sfreq=float(self.SFREQ), l_freq=35., h_freq=50., l_trans_bandwidth=1., h_trans_bandwidth=1.,
                                n_jobs=4, method='fir', phase='zero', copy=True, verbose=False)

            trial_inst_pwr_theta = self.__get_power(theta)
            trial_inst_pwr_alpha = self.__get_power(alpha)
            trial_inst_pwr_beta = self.__get_power(beta)
            trial_inst_pwr_gamma = self.__get_power(gamma)

            feat_contents = {}
            feat_contents['feat_inst_theta'] = trial_inst_pwr_theta
            feat_contents['feat_inst_alpha'] = trial_inst_pwr_alpha
            feat_contents['feat_inst_beta'] = trial_inst_pwr_beta
            feat_contents['feat_inst_gamma'] = trial_inst_pwr_gamma

            feats.append(feat_contents)
        return feats

class FreqPower(MotorTransform):
    """
    Does not collapse the feature across time (see AvgFreqPower)
    """
    data_keys = ["feat_inst_beta"] # ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta", "feat_inst_gamma"]

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
        d = [data[k] for k in self.data_keys] # KEYS SLICE F
        tensor = np.array(d)
        tensor = np.transpose(tensor, (0,2,1)) # SLICE KEYS F
        return tensor

class AvgFreqPower(MotorTransform):
    data_keys = ["feat_inst_beta"] # ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta", "feat_inst_gamma"]

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
        # data is a sample T Ch
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

class VarFreqPower(MotorTransform):
    data_keys = ["feat_inst_beta"] # ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta", "feat_inst_gamma"]

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
        # data is a sample T Ch
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

        d = [np.power(np.std(
                np.take(normalized_data[k], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False), 2) # SLICE F
            for k in self.data_keys] # KEYS SLICE F

        tensor = np.array(d)
        tensor = np.transpose(tensor, (1,0,2)) # SLICE KEYS F
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2])) # SLICE F
        return tensor

class TopomapNN(MotorTransform, NNUtils):
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
                np.ones(64), self.info, ch_type='eeg', sensors=False, contours=0,
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
        # result = feat_tensor
        del feat_tensor # free gpu memory

        result = result.numpy(force=True) # always force back to cpu and np
        return result # T F

class MotorDataset(torch.utils.data.Dataset):
    SFREQ = 160

    KEY_MAP = {
        "real-left-fist":     "TASK1T1",
        "real-right-fist":    "TASK1T2",
        "imagine-left-fist":  "TASK2T1",
        "imagine-right-fist": "TASK2T2",
        "imagine-both-fist":  "TASK4T1",
        "imagine-both-feet":  "TASK4T2"
    }

    RUN_MAP = {
        "real-left-fist":     [3, 7, 11],
        "real-right-fist":    [3, 7, 11],
        "imagine-left-fist":  [4, 8, 12],
        "imagine-right-fist": [4, 8, 12],
        "imagine-both-fist":  [6, 10, 14],
        "imagine-both-feet":  [6, 10, 14]
    }

    def __init__(self,
            data_dir = '/net2/derData/deep-eeg/asr-feats/ds004362',   # location of asr cleaned data bundled with markers and channels
            subjects:list=None,                                       # subjects to use, default to all
            n_sessions=None,                                          # number of sessions to pick, default all
            y_keys=["real-left-fist", "real-right-fist"],             # real-left, real-right
            x_params={
                "feature": "TopomapNN",                               #
                "window": -1,                                         # number of samples to average over (-1: full session)
                "stride": -1,                                         # number of samples to stride window by. Default is trial epoching, no stride (-1)
                "prestim": 0.5,
                "postim": 1.5
            },
            balanced=True,                                           # within k-cv only; enforce class balance y_mode (only for split and bimodal); only on first y_key
            n_cv=None,                                                # (k,folds) Use this to control train vs test; independent of seed
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir

        self.sessions = [i.split('.')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'mat']
        # only keep relevant runs as defined by y_keys
        re_filter = "|".join(list(set([f"run-{i}" for key in y_keys for i in self.RUN_MAP[key]])))
        self.sessions = [session for session in self.sessions if re.search(re_filter, session)]

        if subjects != None:
            self.sessions = [i for i in self.sessions if i.split("-")[1].split("_")[0] in subjects]
        n_sessions = n_sessions if n_sessions is not None else len(self.sessions)
        if n_sessions > len(self.sessions):
            print("Warning: n_sessions cannot be larger than sessions")
        self.sessions = self.sessions[:n_sessions]
        self.subjects = [i.split("-")[1].split("_")[0] for i in self.sessions] # file name format: sub-000_*

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

        for y_key in y_keys:
            if y_key not in MotorDataset.KEY_MAP.keys():
                raise ValueError(f"Invalid y_keys: {y_key}")
        self.y_keys = y_keys
        self.balanced = balanced

        # Preload data
        self.raw_data = [self.__preload_raw(session) for session in tqdm(self.sessions)]
        # Epoching and balancing
        self._marker_names = [self.raw_data[n_session]['evt_markers_names'][0] for n_session in range(len(self.raw_data))]
        self._marker_onsets = [self.raw_data[n_session]['evt_markers_sample'][0] for n_session in range(len(self.raw_data))]

        self.set_x_transformer(x_params)

    def set_x_transformer(self, x_params):
        self.prestim = x_params["prestim"]
        self.postim = x_params["postim"]

        # Instantiate transformer
        if "sfreq" not in x_params:
            x_params['sfreq'] = self.SFREQ
        self.x_params = x_params

        if type(x_params["feature"]) is str:
            try:
                transformer_cls = globals()[x_params["feature"]]
            except KeyError:
                raise ValueError("x_params.feature class not found")
        else:
            transformer_cls = x_params["feature"]
        self.x_transformer = transformer_cls(self.x_params)

        # Transform data (populates self.data and self.ch_names). Note: this modifies input data.
        self.__transform_raw(copy.deepcopy(self.raw_data))

        # Trialize data
        self.__trialize(self.data, self._marker_names, self._marker_onsets)

        # Transform epoched samples to desired feature
        self.data = [self.x_transformer.transform(d) for d in tqdm(self.data)] # S T F ..

    def __preload_raw(self, session):
        mat_data = scipy.io.loadmat(os.path.join(self.basedir, session))
        all_data_keys = ["data", "channames", "evt_markers_names", "evt_markers_sample", 
                        "feat_inst_theta", "feat_inst_alpha", "feat_inst_beta", "feat_inst_gamma"]
        mat_data = {k: v for k,v in mat_data.items() if k in self.y_keys or k in all_data_keys}
        return mat_data

    def __transform_raw(self, data):
        # Transform channels to be consistent across sessions using template in x_transformer
        self.ch_names = self.x_transformer.info['ch_names']
        for n_session in range(len(data)):
            # each session (file) contains multiple epochs data
            session_ch_names = [ch[0].strip() for ch in data[n_session]['channames'][0]] # matlab forces fixed len str
            ch_order = [session_ch_names.index(i) for i in self.ch_names]
            data[n_session]['data'] = data[n_session]['data'][ch_order, :] # Ch T

        # Transform raw to instantaneous spectral power
        self.data = self.x_transformer.get_spectral_power(data)

    def __trialize(self, data, data_markers, data_onsets):
        # data[0][data_keys[0]].shape == ch x time
        # perform epoching and class-balancing
        mapped_y_keys = [self.KEY_MAP[key] for key in self.y_keys]
        prestim_samples = int(self.prestim * self.SFREQ)
        postim_samples = int(self.postim * self.SFREQ)
        epoch_data = []
        skipped_sessions = []
        for n_session in range(len(data)):
            filtered_markers = np.array([marker[0].strip() for marker in data_markers[n_session] if marker in mapped_y_keys])
            onsets = np.array([data_onsets[n_session][n] for n, marker in enumerate(data_markers[n_session]) if marker in mapped_y_keys])
            # get epochs, taking out out-of-bound indices
            if len(filtered_markers) > 0:
                if isinstance(onsets[0], np.floating):
                    skipped_sessions.append(self.sessions[n_session])
                    continue
                # prepare sets of epoch slices
                slices = [list(range(x-prestim_samples,x+postim_samples)) for x in onsets]

                # taking out out-of-bound indices
                inbounds = [e for e, epoch_indices in enumerate(slices) if epoch_indices[0] > 0 and epoch_indices[-1] < data[n_session][self.x_transformer.data_keys[0]].shape[1]]
                slices = np.array(slices)[inbounds]
                filtered_markers = filtered_markers[inbounds]

                # epoching
                for x_key in self.x_transformer.data_keys:
                    data[n_session][x_key] = np.array([data[n_session][x_key][:, s] for s in slices])

                if data[n_session][x_key].shape[0] != len(filtered_markers):
                    raise ValueError(f"Number of epochs and associated markers don't match")

                # epochs is a list of dict
                for s in range(len(filtered_markers)):
                    epoch = {}
                    for x_key in self.x_transformer.data_keys:
                        epoch[x_key] = np.transpose(data[n_session][x_key][s,:,:], (1,0)) # T x CH
                    epoch['onset'] = onsets[s]
                    epoch['marker'] = filtered_markers[s]
                    epoch['subject'] = self.subjects[n_session]
                    epoch_data.append(epoch)

        # flatten the data
        self.data = np.array(epoch_data)
        self.onsets = np.array([epoch['onset'] for epoch in epoch_data])
        self.markers = np.array([epoch['marker'] for epoch in epoch_data])
        self.y_data = np.array([epoch['marker'] for epoch in epoch_data])
        self.subjects = np.array([epoch['subject'] for epoch in epoch_data])

        if len(self.y_data) == 0:
            print("Warning: Empty data. Data doesn't contain desired markers")
        else:
            # Balancing
            self.y_data = np.where(self.y_data == mapped_y_keys[0], 1, 0)
            if self.balanced:
                y_counts = [np.sum(self.y_data == 0), np.sum(self.y_data == 1)]
                remove_idxs = np.argwhere(self.y_data == np.argmax(y_counts))[0:max(y_counts)-min(y_counts)]
                keep_idxs = [idx for idx in range(len(self.y_data)) if idx not in remove_idxs]
                self.y_data = self.y_data[keep_idxs]
                self.data = self.data[keep_idxs]
                self.subjects = self.subjects[keep_idxs]
                self.onsets = self.onsets[keep_idxs]
                self.markers = self.markers[keep_idxs]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y_data[idx]
