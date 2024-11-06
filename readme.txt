This is a demo code of self-calibrating random forest applied in myoelectric control.

run 'pip install -r requirements.txt' to install required packages in your environment

run main_feature_extract.py to extract and save features from EMG signals.

run main_replay_validation.py to implement self-calibrating random forest on data extracted from multiple testing blocks. Self-calibration is performed after each testing block.

If you use our codes, please cite our paper entitled: 'Plug-and-Play Myoelectric Control via a Self-Calibrating Random Forest Common Model'