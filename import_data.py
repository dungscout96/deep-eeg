import pandas as pd
import argparse
import os
import mne
import pickle
import numpy as np
from typing import Tuple
import logging

def get_raw_data(raw_file) -> Tuple[bool, np.array]:
    """
    Extract raw data from associated files

    Parameters:
        path: path to where the files are contained in
        raw_file: raw data file

    Returns:
        A tuple (status, data) containing import status and raw data matrix
    """
    try:
        raw = mne.io.read_raw(raw_file)
    except:
        return False, []
    else:
        data = raw.get_data()
        return True, data


def get_chan_info(chan_file) -> Tuple[bool, list, list]:
    """
    Get channel names and locations

    Parameter:
        chan_file: file containing channel names. It could be .vhdr, .set, or *electrodes.tsv file

    Returns:
        Tuple with 3 elements, in order:
            - channel import status. True if both channel names and locations were extracted. False otherwise
            - List of channel names
            - List of tuples (x, y, z)
    """
    channames = []
    chanlocs = []
    status = True
    if chan_file.endswith("electrodes.tsv"):
        df = pd.read_csv(chan_file, delimiter="\t")
        channames = list(
            df["name"]
        )  # name is a required key in electrodes.tsv hence always exists
        chanlocs = list(zip(df["x"], df["y"], df["z"]))
    else:
        try:
            raw = mne.io.read_raw(chan_file)
        except:
            status = False
        else:
            for chan in raw.info["chs"]:
                channames.append(chan["ch_name"])
                chanlocs.append(chan["loc"][:3])

    return status, channames, chanlocs


def get_eeg_events_markers(event_file) -> Tuple[bool, list]:
    """
    Get list of event markers and their start time and duration in tuple from BIDS events file

    Returns:
        A tuple with two element:
            - Import status (True or False) indicating if event markers were successfully imported
            - List of tuples of (marker_id, start time, duration)
    """
    try:
        evt_tbl = pd.read_csv(event_file, delimiter="\t")
    except:
        return False, []
    else:
        evt_col = ""
        if "HED" in evt_tbl.columns:
            evt_col = "HED"
        elif "value" in evt_tbl.columns:
            evt_col = "value"
        elif "trial_type" in evt_tbl.columns:
            evt_col = "trial_type"
        else:
            return False, []

        evt_list = list(zip(evt_tbl[evt_col], evt_tbl["onset"], evt_tbl["duration"]))

        return True, evt_list


def import_data(
    report_csv,
    outpath,
    debug_ds,
    resume=False,
    get_raw=True,
    get_metadata=True,
) -> None:
    """
    Import raw recordings indexed by report csv and save raw data and
    channel names in a pickle file for each recording.
    Update the report csv file with import status. If both the raw data and
    channel names were extracted successfully, with the same number of channel
    names as first dimension of the raw data matrix, status is True. Otherwise, False

    Parameters:
        report_csv: csv file whose each row contains information about a recording file
        outpath: directory to store the updated report csv file
        debug_ds: dataset ID to use for debugging. If empty, not in debug mode
        resume: if resume from last run. Default False which will run the import for all datasets
        get_raw: whether to import raw data. Default True
        get_metadata: whether to import metadata (channel and event info). Default True
    """
    df = pd.read_csv(report_csv)
    import_report = os.path.join(
        outpath, f"{os.path.splitext(report_csv)[0]}_import_status.csv"
    )

    if resume:
        df_import = pd.read_csv(import_report)
    else:
        pd.DataFrame(
            columns=[
                "meeg_json",
                "subject",
                "import_finished",
                "raw_imported",
                "chanloc_imported",
                "event_imported",
            ]
        )
    for index, row in df.iterrows():
        try:
            if debug_ds:
                if row["ds_uid"] != debug_ds:
                    continue
            if resume:
                if row["meeg_json"] in df_import["meeg_json"].values:
                    continue
            # create output directory if not exist
            dataset_path = os.path.join(outpath, row["ds_uid"])
            os.mkdir(dataset_path, exist_ok=True)

            raw_status = False
            if get_raw:
                # get raw data
                files = [f for f in row["raw_file"].split(",") if ".vhdr" in f]
                if len(files) > 1:
                    logging.warning(f'Multiple .vhdr files found for {row["meeg_json"]} of {row["ds_uid"]}.\nUsing the first file {files[0]}')
                    raw_file = files[0]
                else:
                    raw_file = row["raw_file"]
                raw_status, data = get_raw_data(os.path.join(row["path"], raw_file))
                if raw_status:
                    raw_outfile = os.path.join(
                        dataset_path, f"{row['meeg_json'][:-5]}.pkl"
                    )
                    with open(raw_outfile, "wb") as f:
                        pickle.dump({"data": data}, f)

            chan_status = False
            evt_status = False
            if get_metadata:
                # get channel names and locations
                if not pd.isna(row["eeg_chanloc_file"]):
                    chan_status, channames, chanlocs = get_chan_info(
                        os.path.join(row["path"], row["eeg_chanloc_file"])
                    )
                elif ".edf" in row["raw_file"]:
                    chan_status, channames, chanlocs = get_chan_info(
                        os.path.join(row["path"], row["raw_file"])
                    )
                else:
                    chan_status = False
                    channames = []
                    chanlocs = []

                # get event markers
                basename = row["meeg_json"][: -len("eeg.json")]
                evt_file = os.path.join(row["path"], basename + "events.tsv")
                if os.path.exists(evt_file):
                    evt_status, evt_markers = get_eeg_events_markers(evt_file)

                # save metadata
                metadata = {}
                if chan_status:
                    metadata["channames"] = channames
                    metadata["chanlocs"] = chanlocs
                if evt_status:
                    metadata["evt_markers"] = evt_markers

                meta_outfile = os.path.join(
                    dataset_path, f"{row['meeg_json'][:-5]}_metadata.pkl"
                )
                with open(meta_outfile, "wb") as f:
                    pickle.dump(metadata, f)

            import_finished = True
            df_import.loc[len(df_import.index)] = [
                row["meeg_json"],
                row["subject"],
                import_finished,
                raw_status,
                chan_status,
                evt_status,
            ]

        except Exception as e:
            logging.warning(f'Error processing {row["meeg_json"]} of {row["ds_uid"]}')
            logging.warning(f"\t{e}")
            import_finished = False
            df_import.loc[len(df_import.index)] = [
                row["meeg_json"],
                row["subject"],
                import_finished,
                raw_status,
                chan_status,
                evt_status,
            ]

        with open(import_report, "w") as out:
            df_import.to_csv(out, index=False)

        if debug_ds and len(df_import.index) == 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import scraped data")
    parser.add_argument(
        "--input",
        dest="input",
        required=True,
        help="path to report file to use",
    )
    parser.add_argument(
        "--output",
        dest="output",
        required=False,
        help="Directory to save updated report, default is current directory",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--debug-ds",
        dest="debug_ds",
        required=False,
        default='',
        help="Dataset ID to use for debugging",
    )
    parser.add_argument(
        "--resume",
        required=False,
        dest="resume",
        help="Resume last run or start fresh",
        action="store_true",
    )
    parser.add_argument(
        "--raw",
        required=False,
        dest="raw",
        help="Import raw data",
        action="store_true",
    )
    parser.add_argument(
        "--metadata",
        required=False,
        dest="metadata",
        help="Import metadata",
        action="store_true",
    )
    args = parser.parse_args()
    # Parse args
    report_file = args.input
    result_path = args.output
    import_data(
        report_file,
        result_path,
        args.debug_ds,
        args.resume,
        args.raw,
        args.metadata,
    )
