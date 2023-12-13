import os
from os import walk
from os.path import join
import pandas as pd
import argparse
from typing import List
from typing import Dict
import json
import datetime


def is_meeg_dataset(path) -> bool:
    """
    Detect BIDS M/EEG dataset by checking for *-eeg.json or *-meg.json files recursively.

    Parameters
        path: full path to dataset
    Return
        True or False reflecting if the dataset is M/EEG
    """
    for root, dirnames, filenames in walk(path, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in [".datalad", ".git"]] # in place editing allowed by walk() with topdown=True
        meeg_files = [f for f in filenames if f[-9:] == "_eeg.json" or f[-9:] == "_meg.json"]
        if meeg_files:
            return True
        else:
            for dirname in dirnames:
                if is_meeg_dataset(join(root, dirname)):
                    return True
    return False


def get_meeg_datasets(nemar_path) -> List:
    """
    Get all M/EEG datasets ds_uid from NEMAR disk

    Parameters
        nemar_path: path to nemar storage
    Returns
        meeg_datasets: list of dataset IDs
    """
    meeg_datasets = []
    for folder in os.listdir(nemar_path):
        if folder.startswith("ds") and is_meeg_dataset(join(nemar_path, folder)):
                meeg_datasets.append(folder)
    return meeg_datasets


def get_meeg_json_files(path) -> List:
    """
    Get all m/eeg.json files in the dataset where each file is associated with a recording.

    Parameters:
        path: directory path
    Returns:
        list of tuples (m/eeg filename, m/eeg filepath)
    """
    files = []
    for root, dirnames, filenames in walk(path, topdown=True, followlinks=False):
        if ".datalad" not in root and ".git" not in root:
            files.extend([(f, root) for f in filenames if f.endswith("_eeg.json") or f.endswith("_meg.json")])
    return files


class Recording:
    def __init__(self, path, ds_uid, meeg_json):
        # each recording is associated with a m/eeg.json file in the same directory
        # the m/eeg.json file is assumed to be in sub-* naming scheme to be associated with a valid recording
        self.path = path
        self.meeg_json = meeg_json
        self.ds_uid = ds_uid

        # desired metadata
        self.duration = 0
        self.type = ""
        self.subject = ""
        self.srate = 0
        self.eeg_count = 0
        self.meg_count = 0
        self.task = ""
        self.task_desc = ""
        self.raw_file = ""
        self.eeg_chanloc_file = ""

        # call functions to parse metadata files and get recording info
        self.get_recording_info()


    def __repr__(self) -> str:
        return str(vars(self))


    def get_recording_info(self) -> None:
        result = self.parse_meeg_json()
        if result:
            self.duration = result["duration"]
            self.type = result["type"]
            self.subject = result["subject"]
            self.srate = result["srate"]
            self.eeg_count = result["eeg_count"]
            self.meg_count = result["meg_count"]
            self.task = result["task"]
            self.task_desc = result["task_desc"]
            if result['eeg_count'] <= 0:
                self.eeg_count = self.channel_count(basename=self.meeg_json[0:-9], type='eeg')
            else:
                self.eeg_count = result['eeg_count']
            self.raw_filenames_str = ','.join(self.get_raw_filename())
            self.eeg_chanloc_filename = self.get_eeg_chanloc_filename()


    def parse_meeg_json(self) -> Dict:
        '''
        Parse the m/eeg.json file for:
            - Recording modality (M/EEG)
            - Recording duration
            - Sampling rate
            - EEG channel count
            - MEG channel count
            - Task name
            - Task description
            - Subject ID

        Returns:
            result: dictionary containing information of interest parsed from meeg_json
        '''
        result = {}
        with open(join(self.path, self.meeg_json)) as f:
            data = json.load(f)
            result["type"] = "MEG" if self.meeg_json.endswith("meg.json") else "EEG" # either eeg.json or meg.json
            result["duration"] = data["RecordingDuration"] if "RecordingDuration" in data else 0
            result["srate"] = data["SamplingFrequency"] if "SamplingFrequency" in data else 0
            result["eeg_count"] = data["EEGChannelCount"]if "EEGChannelCount" in data else 0
            result["meg_count"] = data["MEGChannelCount"]if "MEGChannelCount" in data else 0
            result["task"] = data["TaskName"]if "TaskName" in data else ""
            result["task_desc"] = data["TaskDescription"]if "TaskDescription" in data else ""
            # per BIDS naming convention, the file should start with sub-<ID>_*
            # we create a globally unique subject ID by prepending the dataset identifier
            try:
                result["subject"] = f"{self.ds_uid}_{self.meeg_json[:self.meeg_json.index('_')]}"
            except:
                result["subject"] = "unknown"
        return result


    def channel_count(self, basename, type="eeg") -> int:
        '''
        Count channels in channels.tsv that have "EEG" type
        This is performed if no EEGChannelCount was detected in m/eeg.json file
        Parameters:
            path: directory path that contains the channels.tsv file
            basename: basename of file to be appended _channels.tsv
        Return
            count of EEG channels from the _channels.tsv file
        '''
        channel_tsv = f"{basename}_channels.tsv"
        if os.path.isfile(channel_tsv):
            chan_data = pd.read_csv(join(self.path, channel_tsv), sep='\t', header=0)
            chan_data = chan_data[chan_data['type'].lower() == type]
            return len(chan_data)

        return 0


    def get_raw_filename(self) -> list:
        '''
        Get the raw file associated with the recording given the allowed raw file extensions in BIDS

        Returns
            Raw recording file name (comma-separated if multiple)
        '''
        extensions = ['edf', 'vhdr', 'vmrk', 'eeg', 'set', 'bdf'] # https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html#eeg-recording-data
        meg_extensions = ['ctf', 'fif', '4d', 'kit', 'kdf', 'itab'] # https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html#meg-file-formats
        extensions.extend(meg_extensions)
        basename = self.meeg_json[:-len("json")]
        raw_file = [basename + ext for ext in extensions if os.path.isfile(join(self.path, basename + ext))]
        return raw_file 


    def get_eeg_chanloc_filename(self) -> str:
        '''
        Get file containing eeg channel location given assumption that channel location only exists in
        *_electrodes.tsv, *_eeg.vhdr, or *_eeg.set

        Returns
            chanloc_file: string of file name assumed to contain channel location. If none found, return empty string
        '''
        extensions = ['electrodes.tsv', 'eeg.vhdr', 'eeg.set']
        basename = self.meeg_json[:-len("eeg.json")]
        for ex in extensions:
            chanloc_file = basename + ex
            if os.path.isfile(join(self.path, chanloc_file)):
                return chanloc_file

        return ""


def scrape_nemar(nemar_path, result_path, metadata, output_type="csv", debug=False, ds=None) -> None:
    '''
    Scrape NEMAR dataset for metadata of interest
    Save result in the result_path file
    Assumption:
        - Only dataset with EEG channel is included
            - If m/eeg.json file doesn't have EEG channel count, use channels.tsv
            - If channels.tsv have "EEG" type channel, count them
            - There might be datasets without EEGChannelCount in m/eeg.json nor "EEG" channel type in channels.tsv. We will exclude them

    Parameters
        nemar_path: path to NEMAR directory containing all datasets
        result_path: path to save result
        metadata: list of metadata key to extract for each recording
        debug: if in debug mode, run on specified example datasets in ds
        ds: datasets to scan on debug mode
    '''
    headers = ['meeg_json','ds_uid','path']
    headers.extend(metadata)
    df = pd.DataFrame(columns=headers)

    if debug:
        datasets = ds
    else:
        datasets = os.listdir(nemar_path)

    if datasets is not None:
        for folder in datasets:
            if folder.startswith("ds"): # Dataset folder name pattern ds*****
                json_files = get_meeg_json_files(join(nemar_path,folder))
                if json_files: # if has m/eeg.json file, it's MEG or EEG dataset
                    for index, json_file in enumerate(json_files):
                        fname, path = json_file
                        if debug:
                            print("json_file", json_file)
                        # Each m/eeg.json which should correspond to a recording
                        # exception is when it's placed at the top-level. In such case, the file won't start with sub-*
                        if fname.startswith('sub-'):
                            rec = Recording(path=path, ds_uid=folder, meeg_json=fname)
                            # Add only recordings with associated raw files
                            if rec.raw_file:
                                # Only include recording with EEG channels
                                if rec.eeg_count > 0:
                                    if debug:
                                        print(rec)
                                    df.loc[len(df.index)] = [getattr(rec, h) for h in headers]

    timestamp = datetime.datetime.now().isoformat()
    report_fname = f"report_{timestamp}.{output_type}" if os.path.exists(join(result_path,f"report.{output_type}")) else f"report.{output_type}"
    with open(join(result_path, report_fname), "w") as out:
        if debug:
            print("Result file:",join(result_path, report_fname))
        if output_type == "csv":
            df.to_csv(out, index=False)
        elif output_type == "md":
            df.to_markdown(out, index=False)
        elif output_type == "db":
            from sqlalchemy import create_engine
            # https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#connect-strings
            engine = create_engine(f'sqlite:///{out}.db', echo=False)
            df.to_sql('report', con=engine)

# print(get_meeg_datasets())
# meeg_datasets = ['ds000247', 'ds003753', 'ds003420', 'ds004010', 'ds002725', 'ds002550', 'ds002893', 'ds003682', 'ds000248', 'ds003190', 'ds003822', 'ds003670', 'ds003082', 'ds003800', 'ds003846', 'ds004024', 'ds003694', 'ds003518', 'ds002680-copy', 'ds003702', 'ds003574', 'ds003523', 'ds001787', 'ds003505', 'ds003568', 'ds003703', 'ds003608', 'ds003987', 'ds003392', 'ds004011', 'ds003474', 'ds002034', 'ds003509', 'ds004186', 'ds003825', 'ds003516', 'ds003633', 'ds003195', 'ds002908', 'ds002336', 'ds004043', 'ds003517', 'ds003506', 'ds003570', 'ds003490', 'ds002720', 'ds002094', 'ds002833', 'ds002218', 'ds000117', 'ds004117', 'ds002722', 'ds002885', 'ds004019', 'ds003944', 'ds003816', 'ds002598', 'ds004075', 'ds003519', 'ds003458', 'ds002724', 'ds003004', 'ds003104', 'ds002791', 'ds004015', 'ds003634', 'ds002712', 'ds002723', 'ds003751', 'ds000246', 'ds003421', 'ds002680', 'ds003739', 'ds003774', 'ds003194', 'ds003805', 'ds002338', 'ds004022', 'ds003947', 'ds004040', 'ds001785', 'ds003766', 'ds001971', 'ds002718', 'ds003352', 'ds002691', 'ds003690', 'ds002814', 'ds003645', 'ds001784', 'ds003810', 'ds003478', 'ds003969', 'ds004000', 'ds003061', 'ds003638', 'ds001849', 'ds003602', 'ds004018', 'ds003775', 'ds002721', 'ds003655', 'ds003483', 'ds003555', 'ds002778', 'ds003885', 'ds003522', 'ds003710', 'ds003343', 'ds002001', 'ds004067', 'ds003801', 'ds002578', 'ds001810', 'ds003620']
# print(get_meeg_json_files(join('/expanse/projects/nemar/openneuro','ds000247')))
#debug_datasets = ['ds000247', 'ds003753', 'ds003420']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape NEMAR storage for dataset information"
    )
    parser.add_argument(
        "--nemar-path",
        dest="nemar_path",
        required=True,
        help="Path to dataset storage on NEMAR",
        default="/expanse/projects/nemar/openneuro"
    )
    parser.add_argument(
        "--output",
        dest="output",
        required=False,
        help="Directory to save report, default is current directory",
        default=os.getcwd()
    )
    parser.add_argument(
        "--output-format",
        dest="output_format",
        required=False,
        help="Format of the result report, either csv or md. Default is csv",
        default='csv'
    )
    parser.add_argument(
        "--debug",
        required=False,
        dest="debug",
        help="Whether in debug mode",
        action='store_true'
    )
    parser.add_argument(
        "--duration",
        required=False,
        help="Get duration of recording",
        action='store_true'
    )
    parser.add_argument(
        "--type",
        required=False,
        dest="type",
        help="Get modality of recording",
        action='store_true'
    )
    parser.add_argument(
        "--subject",
        required=False,
        dest="subject",
        help="Get subject of recording",
        action='store_true'
    )
    parser.add_argument(
        "--srate",
        required=False,
        dest="srate",
        help="Get sampling rate of recording",
        action='store_true'
    )
    parser.add_argument(
        "--channel-count",
        required=False,
        dest="channel_count",
        help="Get channel counts of recording",
        action='store_true'
    )
    parser.add_argument(
        "--task",
        required=False,
        dest="task",
        help="Get task information of recording",
        action='store_true'
    )
    parser.add_argument(
        "--raw-file",
        required=False,
        dest="raw_file",
        help="Get task information of recording",
        action='store_true'
    )
    parser.add_argument(
        "--chanlocs",
        required=False,
        dest="chanlocs",
        help="Get channel location file of recording",
        action='store_true'
    )

    args = parser.parse_args()
    # Parse args
    nemar_path = args.nemar_path
    result_path = args.output
    output_format = args.output_format

    metadata_list = []
    if args.duration:
        metadata_list.append('duration')
    if args.type:
        metadata_list.append('type')
    if args.srate:
        metadata_list.append('srate')
    if args.subject:
        metadata_list.append('subject')
    if args.channel_count:
        metadata_list.append('eeg_count')
        metadata_list.append('meg_count')
    if args.task:
        metadata_list.append('task')
        metadata_list.append('task_desc')
    if args.raw_file:
        metadata_list.append('self.raw_filenames_str')
    if args.chanlocs:
        metadata_list.append('eeg_chanloc_filename')

    if args.debug:
        ds = ["ds002725", "ds003753", "ds004010", "ds003420"]
        print(f"NEMAR path: {nemar_path}\nResult path: {result_path}\nOutput format: {output_format}")
        print("Metadata list:", metadata_list)
        print("Datasets:", ds)
        scrape_nemar(nemar_path, result_path, metadata_list, output_format, debug=True, ds=ds)
    else:
        scrape_nemar(nemar_path, result_path, metadata_list, output_format)

