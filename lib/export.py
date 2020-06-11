import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

from lib.txtfile import TxtFile

def export(cfg, box_dir):
    ecg_groups = cfg['export']['ecg_groups']
    export_dir = cfg['data']['dataset_path']
    lead_names = cfg['data']['lead_names']

    for ecg_group in ecg_groups:
        # Get the list of text files and matching labels for each export group
        txt_paths = glob(os.path.join(box_dir, ecg_group['path'], "**/*.txt"), recursive=True)
        seglabel_paths = {f: f"{f}.label" for f in txt_paths if os.path.exists(f"{f}.label")}
        print(f"{ecg_group['name']}: Found {len(seglabel_paths)} labelled TXT files of {len(txt_paths)}")
        if not os.path.exists(os.path.join(export_dir, ecg_group['name'])):
            os.makedirs(os.path.join(export_dir, ecg_group['name']))

        # Iterate over each text file and label
        for txt_path, seglabel_path in tqdm(seglabel_paths.items()):
            with open(seglabel_path, "rb") as f:
                seglabel_data = pickle.load(f)
            study_name = os.path.basename(os.path.dirname(seglabel_path))
            source_range_or_marker, source_typ, source_loc = ecg_group['from']

            # Get start time - check exactly 1 starter marker in each file
            starts = [seg for seg in seglabel_data[source_range_or_marker] if seg['type'] == source_typ]
            if len(starts) > 1:
                print(f"WARNING: Found > 1 segmentation label for {source_typ} in {seglabel_path} - Skipping...")
                continue
            if starts:
                start = int(starts[0][source_loc])
            else:
                # print(f"WARNING: Can't find {source_typ} in {seg_file} - Skipping")
                continue  # File doesn't have this entry - Not an error, probably

            # Get ending time
            assert type(ecg_group['to']) == int, "Absolute 'to' lengths only for the moment"
            end = start + ecg_group['to']

            # Load the TxtFile
            try:
                txtfile = TxtFile(filepath=txt_path)
            except ValueError as e:
                print(f"Problem with file {txt_path}, probably duplicate column names? Skipping... ({e})")
                continue

            # If 'recode' needed, add this to the seg_data
            # If a recode is listed, but it can't find the required label, beat is marked as invalud and not extracted
            # This is necessary to stop the pytorch dataloader panicking when certain keys are missing from the dict.
            invalid_due_to_recode_error = False
            if 'recode' in ecg_group:
                for recode_list in ecg_group['recode']:
                    source_range_or_marker, source_typ, source_loc = recode_list[0]
                    recode_target = recode_list[1]
                    recode_sources = [seg for seg in seglabel_data[source_range_or_marker] if seg['type'] == source_typ]
                    if len(recode_sources) != 1:
                        print(f"WARNING: {txt_path} {len(recode_sources)} segmentation labels for {source_typ} in {seglabel_data}, Skipping!")
                        invalid_due_to_recode_error = True
                    else:
                        seglabel_data[recode_target] = recode_sources[0][source_loc]
            if invalid_due_to_recode_error:
                continue

            # Trim the TxtFile from start:end, including any offset
            # Label is embedded, with also the start time of the snip
            try:
                txtfile_filtered = txtfile.data[lead_names].loc[start + ecg_group['offset']:end + ecg_group['offset'] - 1]  # -1 as up to and including row
                exclude_shorter_than = cfg['export']['exclude_shorter_than']
                if exclude_shorter_than:
                    if len(txtfile_filtered) < exclude_shorter_than:
                        print(f"beat from {txt_path} was shorter than {exclude_shorter_than} ({len(txtfile_filtered)}) -> skipping")
                        continue
                np.savez_compressed(
                    os.path.join(export_dir, ecg_group['name'], f"{study_name}_{os.path.basename(txtfile.filepath)}"),
                    data=txtfile_filtered,
                    labels=seglabel_data,
                    starttime=start + ecg_group['offset'])
            except KeyError as e:
                print(f"Skipping {txt_path} as missing ECG lead :{e}")
