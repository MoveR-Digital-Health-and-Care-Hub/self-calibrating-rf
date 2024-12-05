This is a demo code of self-calibrating random forest applied in myoelectric control.

run 'pip install -r requirements.txt' to install required packages in your environment

If you use our codes and data, please cite below papers:

[1] Xinyu Jiang, Chenfei Ma, Kianoush Nazarpour, "Plug-and-play myoelectric control via a self-calibrating random forest common model.", Journal of Neural Engineering, 2024.
[2] Xinyu Jiang, Chenfei Ma, Kianoush Nazarpour, "Posture-invariant myoelectric control with self-calibrating random forests.", Frontiers in Neurorobotics, 2024.

The code package consists of two parts, a demo and a comprehensive validation.



1 The Demo
The demo codes run our self-calibrating random forest proposed in [1] on data from one example participant. The pre-trained model used in this demo was built on data from 38 participants, and self-calibrates after each testing block, as in [1]. To run the demo codes, please follow below instructions:

1.1 run "demo_feature_extract.py" to extract and save features from demo EMG data of the example participant.

1.2 run "demo_replay.py" to implement self-calibrating random forest on features extracted from multiple testing blocks. Self-calibration is performed after each testing block.




2. The comprehensive validation.
In the comprehensive validation, we mixed all data used in [1], with a total of 66 participants, to pre-train a more powerful random forest model. The model was then applied to data from 20 participants with different arm positions, and self-calibrates to adapt to new arm positions, as in [2]. To run the codes for a comprehensive validation, please follow below instructions:

2.1 run "main_feature_extract_arm_position.py" to extract and save features from EMG data of 20 participants.

2.2 run "main_replay_arm_position.py" to implement self-calibrating random forest on features extracted from multiple testing blocks. Self-calibration is performed after each testing block.




3. Dataset descriptions

3.1 The demo dataset (in the folder "dataset_demo") is from one participant, with a calibration session and a testing session (10 testing blocks, blocks 0-4 on day 1 and blocks 5-9 on day 2). For each trial, participants were asked to hold a constant hand gesture in the last 1 second.

3.2 The arm position dataset (in the folder "dataset_arm_position") is from 20 new participants, with a calibration session and a testing session (11 testing blocks, collected on the same day) for each participant. We defined 5 different arm positions, denoted as P2, P4, P5, P6 and P8, as defined in [2]. Data in the calibration session were collected with participants' elbow positioned at a angle of 90 degrees (P5). The arm positions in the testing session were fixed in a single block but varied in different blocks. Participants 0-9 are allocated into group A and participants 10-19 are allocated into group B. The arm position sequence for group A in 11 testing blocks is: P5-P4-P2-P6-P8-P5-P8-P6-P2-P4-P5. The arm position sequence for group B in 11 testing blocks is: P5-P8-P6-P2-P4-P5-P4-P2-P6-P8-P5. For more details, please refer to [2]. The sampling rate for both datasets is 2000 Hz. Data in both datasets are 8-channel EMG.



