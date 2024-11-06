import numpy as np
import emg_functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import copy
import os
import joblib

main_path='./'
data_path = main_path + '/dataset'
feature_path = main_path + '/features'
results_path = main_path + '/results'
if not os.path.exists(results_path):
    os.makedirs(results_path)

nTree=400
nTreeUpdate=80
fs=2000
Nchannel=8
ssc_thresh=0
cal_repetition=1
label_select=[0,1,2,3,4,5]
max_example=0.07
max_depth = None
max_sample_buffer=30*10*5
min_sample_class=10
method='kmeans'

Nsubject=1
Nrepeat=1

for repeat in range(0,Nrepeat):

    Ntesting_blocks = 10
    accuracy_self_calibration = np.zeros([Nsubject, Ntesting_blocks])
    accuracy_pretrain = np.zeros([Nsubject, Ntesting_blocks])
    accuracy_prune_append = np.zeros([Nsubject, Ntesting_blocks])
    accuracy_standard_RF = np.zeros([Nsubject, Ntesting_blocks])
    accuracy_standard_LDA = np.zeros([Nsubject, Ntesting_blocks])



    feature_dataset3_calibration_all = list()
    label_dataset3_calibration_all = list()
    label_dataset3_calibration_subject_all = list()

    feature_dataset3_test_all = list()
    label_dataset3_test_all = list()
    label_dataset3_test_subject_all = list()
    label_dataset3_test_block_all = list()

    feature_calibration_all = list()
    label_calibration_all = list()
    label_calibration_subject_all = list()

    feature_test_all = list()
    label_test_all = list()
    label_test_subject_all = list()



    for subject in range(Nsubject):


        folder = 'calibration'
        label = np.load(data_path + '/' + str(subject) + '/' + folder + '/label.npy')

        feature_mav = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_mav.npy')
        feature_wl = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_wl.npy')
        feature_ssc = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_ssc.npy')
        feature_zc = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_zc.npy')
        feature_rms = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_rms.npy')
        feature_skw = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_skw.npy')
        feature_mnf = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_mnf.npy')
        feature_mdf = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_mdf.npy')
        feature_pkf = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_pkf.npy')
        feature_vcf = np.load(feature_path + '/' + str(subject) + '/' + folder + '/feature_vcf.npy')

        feature_mav_reshape = np.reshape(feature_mav, (-1, 8), 'C')
        feature_wl_reshape = np.reshape(feature_wl, (-1, 8), 'C')
        feature_ssc_reshape = np.reshape(feature_ssc, (-1, 8), 'C')
        feature_zc_reshape = np.reshape(feature_zc, (-1, 8), 'C')
        feature_rms_reshape = np.reshape(feature_rms, (-1, 8), 'C')
        feature_skw_reshape = np.reshape(feature_skw, (-1, 8), 'C')
        feature_mnf_reshape = np.reshape(feature_mnf, (-1, 8), 'C')
        feature_mdf_reshape = np.reshape(feature_mdf, (-1, 8), 'C')
        feature_pkf_reshape = np.reshape(feature_pkf, (-1, 8), 'C')
        feature_vcf_reshape = np.reshape(feature_vcf, (-1, 8), 'C')

        Ntrial = len(label)
        Nwindow = np.size(feature_mav, axis=1)
        label_calibration = np.repeat(label, Nwindow)
        feature_calibration = np.concatenate((feature_rms_reshape, feature_wl_reshape, feature_ssc_reshape, feature_zc_reshape,
                                  feature_mav_reshape, feature_skw_reshape, feature_vcf_reshape, feature_mdf_reshape,
                                  feature_mnf_reshape, feature_pkf_reshape), axis=1)


        mean_val_cal = np.mean(feature_calibration, axis=0)
        std_val_cal = np.std(feature_calibration, axis=0)
        feature_calibration_norm = (feature_calibration - mean_val_cal) / std_val_cal

        mdl_append = RandomForestClassifier(n_estimators=int(nTree / 2))
        mdl_append.fit(feature_calibration_norm, label_calibration)
        mdl_append = emg_functions.get_sklearn_rf_parameters(mdl_append)

        mdl_rf = RandomForestClassifier(n_estimators=int(nTree))
        mdl_rf.fit(feature_calibration_norm, label_calibration)
        mdl_rf = emg_functions.get_sklearn_rf_parameters(mdl_rf)

        mdl_lda = LinearDiscriminantAnalysis()
        mdl_lda.fit(feature_calibration_norm, label_calibration)


        mdl_pretrain = joblib.load(main_path + 'model_pretrain/model_pretrain.joblib')
        mdl_pretrain = emg_functions.get_sklearn_rf_parameters(mdl_pretrain)

        mdl_prune = emg_functions.prune(copy.deepcopy(mdl_pretrain), feature_calibration_norm, label_calibration)
        mdl_prune_append = emg_functions.append(copy.deepcopy(mdl_prune), copy.deepcopy(mdl_append))

        mdl_self_calibration = copy.deepcopy(mdl_prune_append)
        mdl_without_clustering = copy.deepcopy(mdl_prune_append)
        mdl_without_tsne = copy.deepcopy(mdl_prune_append)

        feature_buffer = np.zeros((0, np.size(feature_calibration_norm, axis=1)))
        feature_buffer_without_clustering = np.zeros((0, np.size(feature_calibration_norm, axis=1)))
        feature_buffer_without_tsne = np.zeros((0, np.size(feature_calibration_norm, axis=1)))

        label_buffer = np.zeros((0,))
        label_buffer_without_clustering = np.zeros((0,))
        label_buffer_without_tsne = np.zeros((0,))

        label_buffer_truth = np.zeros((0,))
        label_buffer_without_clustering_truth = np.zeros((0,))
        label_buffer_without_tsne_truth = np.zeros((0,))

        for idx_test_block in range(10):

            label = np.load(data_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/label.npy')
            feature_mav = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_mav.npy')
            feature_wl = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_wl.npy')
            feature_ssc = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_ssc.npy')
            feature_zc = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_zc.npy')
            feature_rms = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_rms.npy')
            feature_skw = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_skw.npy')
            feature_mnf = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_mnf.npy')
            feature_mdf = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_mdf.npy')
            feature_pkf = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_pkf.npy')
            feature_vcf = np.load(feature_path + '/' + str(subject) + '/testing_block' + str(idx_test_block) + '/feature_vcf.npy')

            feature_mav_reshape = np.reshape(feature_mav, (-1, 8), 'C')
            feature_wl_reshape = np.reshape(feature_wl, (-1, 8), 'C')
            feature_ssc_reshape = np.reshape(feature_ssc, (-1, 8), 'C')
            feature_zc_reshape = np.reshape(feature_zc, (-1, 8), 'C')
            feature_rms_reshape = np.reshape(feature_rms, (-1, 8), 'C')
            feature_skw_reshape = np.reshape(feature_skw, (-1, 8), 'C')
            feature_mnf_reshape = np.reshape(feature_mnf, (-1, 8), 'C')
            feature_mdf_reshape = np.reshape(feature_mdf, (-1, 8), 'C')
            feature_pkf_reshape = np.reshape(feature_pkf, (-1, 8), 'C')
            feature_vcf_reshape = np.reshape(feature_vcf, (-1, 8), 'C')

            Ntrial = len(label)
            Nwindow = np.size(feature_mav, axis=1)
            label_block = np.repeat(label, Nwindow)
            feature_block = np.concatenate((feature_rms_reshape, feature_wl_reshape, feature_ssc_reshape, feature_zc_reshape,
                                      feature_mav_reshape, feature_skw_reshape, feature_vcf_reshape,
                                      feature_mdf_reshape,
                                      feature_mnf_reshape, feature_pkf_reshape), axis=1)

            num_sample = len(label_block)

            feature_block_norm = (feature_block - mean_val_cal) / std_val_cal

            label_pretrain, _ = emg_functions.get_predictions(mdl_pretrain, feature_block_norm)
            accuracy_pretrain[subject, idx_test_block] = np.mean(label_pretrain == label_block)

            label_prune_append, _ = emg_functions.get_predictions(mdl_prune_append, feature_block_norm)
            accuracy_prune_append[subject, idx_test_block] = np.mean(label_prune_append == label_block)

            label_self_calibration, _ = emg_functions.get_predictions(mdl_self_calibration, feature_block_norm)
            accuracy_self_calibration[subject, idx_test_block] = np.mean(label_self_calibration == label_block)

            label_rf, _ = emg_functions.get_predictions(mdl_rf, feature_block_norm)
            accuracy_standard_RF[subject, idx_test_block] = np.mean(label_rf == label_block)

            label_lda = mdl_lda.predict(feature_block_norm)
            accuracy_standard_LDA[subject, idx_test_block] = np.mean(label_lda == label_block)

            feature_buffer, label_buffer = emg_functions.buffer_update(copy.deepcopy(max_sample_buffer), copy.deepcopy(feature_buffer), copy.deepcopy(label_buffer),
                                                                       copy.deepcopy(feature_block_norm), copy.deepcopy(label_self_calibration))


            feature_select, label_pseudo, label_buffer = emg_functions.manifold_clustering(
                copy.deepcopy(feature_buffer), copy.deepcopy(label_select), copy.deepcopy(min_sample_class), copy.deepcopy(mdl_prune_append), method,
                embedding='tsne')


            label_mixed_block = np.concatenate((label_pseudo, label_calibration), axis=0)
            feature_mixed_block_norm = np.concatenate((feature_select, feature_calibration_norm), axis=0)
            mdl_replace = RandomForestClassifier(n_estimators=nTreeUpdate)
            mdl_replace.fit(feature_mixed_block_norm, label_mixed_block)
            mdl_replace = emg_functions.get_sklearn_rf_parameters(copy.deepcopy(mdl_replace))
            idx_remain_tree = np.concatenate((np.array(range(0, int(nTree / 2))),
                                              np.random.permutation(range(int(nTree / 2), nTree))[
                                              :(int(nTree / 2 - nTreeUpdate))]), axis=0)
            mdl_self_calibration = emg_functions.replace(copy.deepcopy(mdl_self_calibration), copy.deepcopy(mdl_replace),
                                                          idx_remain_tree)


        print('Accuracy of LDA in 10 testing blocks')
        print(np.mean(accuracy_standard_LDA[0:subject + 1, :], axis=0))
        print('Accuracy of RF in 10 testing blocks')
        print(np.mean(accuracy_standard_RF[0:subject + 1, :], axis=0))
        print('Accuracy of pre-trained RF in 10 testing blocks')
        print(np.mean(accuracy_pretrain[0:subject + 1, :], axis=0))
        print('Accuracy of pre-trained and fine-tuned RF in 10 testing blocks')
        print(np.mean(accuracy_prune_append[0:subject + 1, :], axis=0))
        print('Accuracy of self-calibrating RF in 10 testing blocks')
        print(np.mean(accuracy_self_calibration[0:subject + 1, :], axis=0))

        np.save(results_path + '/accuracy_realtime_re_analysis_standard_LDA_repeat' + str(repeat) + '.npy', accuracy_standard_LDA)
        np.save(results_path + '/accuracy_realtime_re_analysis_standard_RF_repeat' + str(repeat) + '.npy', accuracy_standard_RF)
        np.save(results_path + '/accuracy_realtime_re_analysis_prune_append_repeat' + str(repeat) + '.npy', accuracy_prune_append)
        np.save(results_path + '/accuracy_realtime_re_analysis_self_calibration_repeat' + str(repeat) + '.npy', accuracy_self_calibration)


print('Competed.')

