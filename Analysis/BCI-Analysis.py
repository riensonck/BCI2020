# -*- coding: utf-8 -*-

"""
Author: Rien Sonck
Date: 2020-05-23
Description: This script belongs to my CAED2020 EEG project for the University of Ghent.
			 This script compares the performance of different classifiers on a motor imagery task.
"""

######################
# Importing packages #
######################
import os
import re
import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, cross_validate, cross_val_predict
from sklearn.metrics import recall_score, confusion_matrix, plot_confusion_matrix, accuracy_score, precision_score, average_precision_score, plot_precision_recall_curve, roc_curve, auc, plot_roc_curve
# sklearn classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns # Need to downgrade matplotlib to 3.1.0 (pip install matplotlib==3.1.0), the newer version results in a conflict with matplotlib. 
from scipy.stats import friedmanchisquare
import scipy.stats as stats
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

#####################
# Default Settings  #
####################

plt.ioff() # turn off interactive plotting
np.random.seed(42) # setting the seed for reproducible results. 

plt.rc('font', size=40)         # controls default text sizes
plt.rc('axes', titlesize=30)    # fontsize of the axes title
plt.rc('axes', labelsize=35)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)   # fontsize of the tick labels
plt.rc('ytick', labelsize=30)   # fontsize of the tick labels
plt.rc('legend', fontsize=30)   # legend fontsize
plt.rc('figure', titlesize=40)  # fontsize of the figure title

#################
# Definitions  #
################

def plot_violins(classifier_results, PATH):
	"""
	@param classifier_results: 	pandas dataframe containing the results.
	@param PATH: string containing the path location of where to save figures. 

	This function plots the average accuracy of each classifier across subjects. 
	The figures are designed that they can fit inside a Latex document without having to edit them. 
	"""

	classifiers = [('LDA','KN'), ('DTL','NB'), ('LR', 'SVC')]
	index_slice = [(0,2), (2,4), (4,6)]

	for classifier_num in range(len(classifiers)):
		# k-fold
		kf_cols = [col for col in classifier_results.columns if re.search('(?=kf.*)(?=.*acc)', col)]  # searches for all column names with the the pattern "kf .... accuracy"
		kf_cols = kf_cols[index_slice[classifier_num][0]:index_slice[classifier_num][1]]  
		kf_subset = classifier_results[kf_cols]
		kf_mean = np.round(np.mean(kf_subset),2)
		# monte carlo
		mc_cols = [col for col in classifier_results.columns if re.search('(?=mc.*)(?=.*acc)', col)]  # searches for all column names with the the pattern "mc .... accuracy"
		mc_cols = mc_cols[index_slice[classifier_num][0]:index_slice[classifier_num][1]]
		mc_subset = classifier_results[mc_cols]
		mc_mean = np.round(np.mean(mc_subset),2)

		# positions 
		pos_kf = [1, 2.5] # [1,4,7,10,13,16]
		pos_mc = [1.5,3] # [2,5,8,11,14,17]

		# Setting up plotting canvas
		fig, ax = plt.subplots()

		#plotting the chance level line
		ax.axhline(0.5, linewidth=1, linestyle= '--')
		# place a text box in upper left in axes coords
		ax.text(1.9,0.5, "Chance level", fontsize = 22,  
			bbox=dict(facecolor='lightblue', alpha=0.5))

		for i in range(len(kf_mean)):
			ax.text(pos_kf[i]-0.05, kf_mean[i] + 0.01, "{0}%".format(kf_mean[i]), fontsize = 22, fontdict=dict(fontweight='bold'))
			ax.text(pos_mc[i]-0.05, mc_mean[i] + 0.01, "{0}%".format(mc_mean[i]), fontsize = 22, fontdict=dict(fontweight='bold'))

		kf_bp = ax.violinplot(np.array(kf_subset), widths = 0.4, vert = True, 
			positions=pos_kf,showmeans=True, showextrema=True)

		mc_bp = ax.violinplot(np.array(mc_subset), widths = 0.3, vert = True, 
			positions=pos_mc,  showmeans=True, showextrema=True)

		ax.legend([kf_bp['bodies'][0], mc_bp['bodies'][0]], 
			['Kfold cross-validation', 'monte-carlo cross-validation'],
			loc='lower right')

		ax.set_ylim(0,1)
		ax.set_xticks([1.25, 2.75]) # [1.5, 4.5, 7.5, 10.5, 13.5, 16.5]
		ax.set_ylabel("Accuracy (%)")
		ax.set_xticklabels([classifiers[classifier_num][0], classifiers[classifier_num][1]]) # 'Linear Discriminant Analysis','KNeighbors' 'Decision Tree Learning' 'Naive Bayes', 'Logistic Regression', 'Support Vector Classification'
		fig.set_figheight(8)
		fig.set_figwidth(25)
		fig.tight_layout() 
		fig.savefig("{0}classifiers_accuracy_{1}_{2}.jpg".format(PATH, classifiers[classifier_num][0], classifiers[classifier_num][1]), dpi = 100)
		# Closing the plots
		fig.clear()
		plt.close(fig)


def plot_cmats(classifier_results, PATH, subject = None):
	"""
	@param classifier_results: pandas dataframe containing the results.
	@param PATH: string containing the path location of where to save figures.
	@param subject: integer, if no subject numer is given, the confusion matrix across all subject is plotted.  

	This function plots the confusion matrix of each classifier across subjects or for an individual subject 
	The figures are designed that they can fit inside a Latex document without having to edit them. 
	"""

	cols = [col for col in classifier_results.columns if re.search('(?=kf|mc.*)(?=.*cmat)', col)]  # searches for all column names with the the pattern "kf .... accuracy"
	mcc_cols = [col for col in classifier_results.columns if re.search('(?=kf|mc.*)(?=.*mcc)', col)]
	subset = classifier_results[cols]
	mcc_subset = classifier_results[mcc_cols]
	labels = [["Left-hand", "Right-hand"], ['LDA','KN','DTL','NB', 'LR', 'SVC']]

	# selecting the confusion matrix and normalizing it
	norm = []
	mcc = []
	for col_num in range(len(cols)):
		if subject != None: # only one subject
			norm.append(np.true_divide(subset[cols[col_num]][subject - 1], subset[cols[col_num]][subject - 1].sum(axis=1, keepdims=True)))
			mcc.append(np.round(mcc_subset[mcc_cols[col_num]][subject - 1],2))
		else: # across subjects
			cmat_sum = np.sum(subset[cols[col_num]])
			norm.append(np.true_divide(cmat_sum, cmat_sum.sum(axis=1, keepdims=True)))
	if subject == None:
		mcc = np.round(np.mean(mcc_subset), 2)

	# Plotting
	fig_kf, axs_kf = plt.subplots(3, 2)
	fig_mc, axs_mc = plt.subplots(3, 2)

	index = np.linspace(0, 10, 6, dtype = int) 

	for i in range(len(axs_kf.reshape(-1))): 
		sns.heatmap(norm[index[i]], annot=True, annot_kws={"size": 22}, vmin = 0, vmax = 1, cmap = plt.cm.Blues,  ax = axs_kf.reshape(-1)[i], 
			xticklabels = labels[0], yticklabels = labels[0]).set_title('{0} (MCC = {1})'.format(labels[1][i], mcc[index[i]]))
		sns.heatmap(norm[index[i] + 1], annot=True, annot_kws={"size": 22}, vmin = 0, vmax = 1, cmap = plt.cm.Blues,  ax = axs_mc.reshape(-1)[i], 
			xticklabels = labels[0], yticklabels = labels[0]).set_title('{0} (MCC = {1})'.format(labels[1][i],mcc[index[i] + 1]))


	for col in range(len(axs_kf[0])):
		for row in range(len(axs_kf)):
			axs_kf[row,col].set_ylabel("True label")
			axs_kf[row,col].set_xlabel("Predicted label")
			plt.setp(axs_kf[row,col].get_xticklabels(), rotation=0)
			axs_mc[row,col].set_ylabel("True label")
			axs_mc[row,col].set_xlabel("Predicted label")
			plt.setp(axs_mc[row,col].get_xticklabels(), rotation=0)

	fig_kf.set_figheight(28)
	fig_kf.set_figwidth(25)
	fig_mc.set_figheight(28)
	fig_mc.set_figwidth(25)
	fig_kf.tight_layout(pad=2.0)
	fig_mc.tight_layout(pad=2.0)
	if subject == None:
		fig_kf.savefig("{0}classifiers_cmat_kf.jpg".format(PATH), dpi = 100)
		fig_mc.savefig("{0}classifiers_cmat_mc.jpg".format(PATH), dpi = 100)
	else:
		fig_kf.savefig("{0}cmat_kf_subject{1}.jpg".format(PATH, subject), dpi = 100)
		fig_mc.savefig("{0}cmat_mc_subject{1}.jpg".format(PATH, subject), dpi = 100)
	# Closing the plots
	fig_kf.clear()
	fig_mc.clear()
	plt.close(fig_kf)
	plt.close(fig_mc)


def plot_roc(classifier_results, PATH, subject, cross_val):
	"""
	@param classifier_results: pandas dataframe containing the results.
	@param PATH: string containing the path location of where to save figures.
	@param subject: integer, if no subject numer is given, the confusion matrix across all subject is plotted.
	@param cross_val: string, either 'kf' for kfold cross-validation ot 'mc' for monte-carlo cross-validation 

	This function plots the confusion matrix of each classifier across subjects or for an individual subject 
	The figures are designed that they can fit inside a Latex document without having to edit them. 
	"""

	index_slice = [(0,2), (2,4), (4,6)]
	subject = subject - 1
	classes = ['LH', 'RH']
	labels = ["Left-hand", "Right-hand"]
	classifiers = [('LDA','KN'), ('DTL','NB'), ('LR', 'SVC')]

	for classifier_num in range(len(classifiers)):
		cols = [col for col in classifier_results.columns if re.search('(?={0}.*)(?=.*roc)'.format(cross_val), col)]  # searches for all column names with the the pattern "kf .... accuracy"
		cols = cols[index_slice[classifier_num][0]:index_slice[classifier_num][1]] # selecting the right index slice for the classifiers that we need
		subset = classifier_results[cols]
		fig, ax = plt.subplots(nrows = 2, ncols = 2)

		for col_num in range(len(cols)):

			aucs = subset[cols[col_num]][subject]['aucs']
			tprs_lh = subset[cols[col_num]][subject]['tprs_lh'] # tprs for each fold for the LH
			tprs_rh = subset[cols[col_num]][subject]['tprs_rh'] # tprs for each fold for the LH
			mean_fpr = subset[cols[col_num]][subject]['mean_fpr']

			tprs = [tprs_lh, tprs_rh]
			# Initialising variables
			mean_tpr, mean_auc, std_tpr, std_auc, tprs_upper, tprs_lower = [], [], [], [], [], []

			for i in range(len(classes)): 
				# calculating mean True Positive Rates, across folds
				mean_tpr.append(np.mean(tprs[i], axis = 0))
				mean_tpr[i][-1] = 1.0
				# Calculating the mean Area Under the Curve (AUC) for each class	
				mean_auc.append(auc(mean_fpr, mean_tpr[i]))
				# Calculating the standard deviations
				std_tpr.append(np.std(tprs[i], axis=0))
				std_auc.append(np.std(pd.DataFrame(aucs)[classes[i]]))
				tprs_upper.append(np.minimum(mean_tpr[i] + std_tpr[i], 1))
				tprs_lower.append(np.maximum(mean_tpr[i] - std_tpr[i], 0)) 

			# Plotting
			colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

			for i in range(len(tprs)):
				for k in range(len(tprs[j])):
					ax[col_num, i].plot(name='ROC fold {0}'.format(k), alpha=0.3, linewidth=3, fontsize = 25)
					ax[col_num, i].plot(mean_fpr, tprs[i][k], linewidth = 3, linestyle='dashdot', label=r'ROC Fold %d (AUC = %0.2f)' % (k + 1, aucs[k]['LH']), color=colors[k], alpha=.5) # label=r'ROC Fold %d (AUC = %0.2f)' % (k + 1, aucs[k]['LH'])

				ax[col_num, i].plot([0, 1], [0, 1], linestyle='--', linewidth=8, color='r', label='Chance', alpha=.8) # plotting chance level
				ax[col_num, i].plot(mean_fpr, mean_tpr[i], color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc[i], std_auc[i]), linewidth=8, alpha=.8) # plotting the mean roc
				ax[col_num, i].fill_between(mean_fpr, tprs_lower[i], tprs_upper[i], color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
				ax[col_num, i].legend(loc='best', fancybox=True) # bbox_to_anchor=(1.25, 0.5)

			ax[col_num,0].set_ylabel("True Negative Rate")
			ax[col_num,0].set_xlabel("False Negative Rate")
			ax[col_num,0].set_title(classifiers[classifier_num][col_num] + ": Left-Hand", fontsize = 40)
			ax[col_num,0].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
			ax[col_num,1].set_ylabel("True Positive Rate")
			ax[col_num,1].set_xlabel("False Positive Rate")
			ax[col_num,1].set_title(classifiers[classifier_num][col_num] + ": Right-Hand", fontsize = 40)
			ax[col_num,1].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
		fig.set_figheight(38)
		fig.set_figwidth(42)
		fig.tight_layout(w_pad = 2)
		fig.savefig("{0}roc_{1}_{2}-{3}_subject{4}.jpg".format(PATH, cross_val, classifiers[classifier_num][0], classifiers[classifier_num][1], subject + 1), dpi = 100)
		# close the plots
		fig.clear()
		plt.close(fig)

def stat_tests(classifier_results):
	"""
	@param classifier_results: pandas dataframe containing the results
	This function prints the output of the stastical analysis to your ipython interface. 
	"""
	for cv in ['kf', 'mc']:
		cols = [col for col in classifier_results.columns if re.search('(?={0}.*)(?=.*acc)'.format(cv), col)]  # searches for all column names with the the pattern "kf .... accuracy"
		subset = classifier_results[cols]

		# Friedman's Test
		stat, p = friedmanchisquare(subset['lda-{0}-acc'.format(cv)], 
				subset['knc-{0}-acc'.format(cv)], 
				subset['dtc-{0}-acc'.format(cv)], 
				subset['gnb-{0}-acc'.format(cv)], 
				subset['svc-{0}-acc'.format(cv)], 
				subset['lor-{0}-acc'.format(cv)])
		print("{0}, {1}".format(stat, p))
		# Holm's test
		if p < 0.05:
			stacked_data = subset.stack().reset_index() 
			stacked_data = stacked_data.rename(columns={'level_0': 'subject','level_1': 'classifier', 0:'accuracy'})
			# Set up the data for comparison (creates a specialised object)
			MultiComp = MultiComparison(stacked_data['accuracy'],stacked_data['classifier']) 
			comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
			print (comp[0])

########################
# Loading in Raw Data  #
########################
# Source: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html
# run 1: baseline, eyes open
# run 2: baseline, eyes closed
# run 3, 7, 11: Motor execution left vs right hand
# run 4, 8, 12: Motor imagery: left vs right hand
# run 5, 9, 13: Motor execution hands vs feet
# run 6, 10, 14: Motor imagery hands vs feet

PATH = os.getenv('HOME') + '/repos/github_CAED/datasets' # path to where to download the dataset
subjects = np.linspace(1,109, 109)  # the subjects to use data from
runs = [4, 8, 12]
rows_list = []

for subject in subjects: 
	fnames = eegbci.load_data(int(subject), runs, PATH) # returns filenames (strings) associated with each subject run
	raws = [mne.io.read_raw_edf(f, preload = True) for f in fnames] # read in the raw data for each file and keep them in memory (preload = True) 
	raw = mne.io.concatenate_raws(raws) # concatentate the raw files together
	# EEG sensor configuration 
	eegbci.standardize(raw) # set the channel names to a standard channel name, removing dots at the end of the names
	montage = mne.channels.make_standard_montage('standard_1005') # Electrodes are named and positioned according to the international 10-05 system
	raw.set_montage(montage) # Set EEG sensor configuration and head digitization

	#################
	# Preprocessing #
	#################
	# Altough Finite Impulse Response (FIR) filters have better attenuation capabilities, I'm using the Butterworth Infinte Impulse Response (IIR) filter.
	# IIR filters need less associated computational power (Iturrate et al., 2020).
	raw.filter(7., 30., method = 'iir', skip_by_annotation='edge') # for motor imaginery we are interested in the frequency range 7-30Hz
	event_id = dict(left_hands= 2, right_hands= 3)
	events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3)) # only selecting the left hand and right hand events
	picks = mne.pick_types(raw.info, eeg = True, stim = False, exclude = 'bads')

	######################
	# Feature Extraction #
	######################
	csp = mne.decoding.CSP(n_components=10, reg=None, log=True, norm_trace=False)

	#####################################
	# Classifier Calibration / Training #
	#####################################
	tmin, tmax = -1., 4. # tmin: Start time before event, tmax: the end time after event
	epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj = True, picks = picks, baseline = None, preload = True)
	epochs_train = epochs.copy().crop(tmin = 1., tmax = 2.) # avoid classification of evoked responses by using epochs that start 1s after cue onset
	X = epochs_train.get_data() # training set
	y = epochs_train.events[:,2] # training targets,  2 = left hand event, 3 = right hand event
	class_balance = np.mean(y == y[0]) # calculates the balance between 2 and 3 targets
	class_balance = max(class_balance, 1. - class_balance) 
	
	# Cross-Validation, helps for assessing how our classifier results will generalize to an independent data set
	kf = KFold(n_splits = 10) # k-fold cross-validation
	mc = ShuffleSplit(10, test_size=0.2) # monte-carlo cross-validation
	cross_validations = [kf, mc] 

	# Choosen classifiers are the same as (Xygonakis et al., 2019)
	# LDA doesn't have a random_state to use, so random state is not used for any classifier
	lda = LinearDiscriminantAnalysis() 
	knc = KNeighborsClassifier()
	dtc = DecisionTreeClassifier() 
	gnb = GaussianNB()
	lor = LogisticRegression()
	svc = SVC(probability = True) #  random_state = 42

	# Pipelines
	lda_pipe = Pipeline([('CSP', csp), ('LDA', lda)])
	knc_pipe = Pipeline([('CSP', csp), ('KNC', knc)])
	dtc_pipe = Pipeline([('CSP', csp), ('DTC', dtc)])
	gnb_pipe = Pipeline([('CSP', csp), ('GNB', gnb)])
	svc_pipe = Pipeline([('CSP', csp), ('SVC', svc)])
	lor_pipe = Pipeline([('CSP', csp), ('LOR', lor)])
	pipelines = [lda_pipe, knc_pipe, dtc_pipe, gnb_pipe, svc_pipe, lor_pipe]

	row = {'subject': subject, 'Chance level': np.round(class_balance, 2),}
	tags = [["lda", "knc", "dtc", "gnb", "svc", "lor"], ["-kf-", "-mc-"]]
	class_names = ["LH", "RH"]

	#######################
	# Classifier Decoding #
	#######################

	for i in range(len(pipelines)):
		for j in range(len(cross_validations)):

			# Initialising variables
			y_score_rh, y_score_lh, y_pred, y_test, tprs_lh, tprs_rh, aucs = [], [], [], [], [], [], []
			mean_fpr = np.linspace(0, 1, 100)

			# Looping through the cross-validation folds 
			for k, (train_index, test_index) in enumerate(cross_validations[j].split(X, y)):
				# within each fold: train and test the model
				fit = pipelines[i].fit(X[train_index], y[train_index])	# Fit boundary line that will divide the left-hand and right-hand class. From now on, one class will be associated with positive scores, the other with negative scores 
				if i != 0 or i !=5: # KNC, DTC, GNB, LOR
					y_score_fold = fit.predict_proba(X[test_index])		# predicted scores for each epoch, which is used to predict to which class the epochs belongs (negative score - LH, positive score - RH)
					y_score_rh_fold = y_score_fold[:, 1]				# score for the positive class (Right-Hand)
					y_score_lh_fold = y_score_fold[:, 0]				# score for the negative class (Left-Hand)
				else: # LDA, SVC
					y_score_rh_fold = fit.decision_function(X[test_index]) # predicted scores for each epoch, which is used to predict to which class the epochs belongs (negative score - LH, positive score - RH)
					y_score_lh_fold = y_score_rh_fold * -1
				y_pred_fold = fit.predict(X[test_index])				# predicted class for each epoch ( 2 = Left-hand class, 3 = right-hand class)
				y_test_fold= y[test_index]
				X_test_fold = X[test_index]

				# across folds: keeping track
				y_score_rh = y_score_rh + list(y_score_rh_fold)
				y_score_lh = y_score_lh + list(y_score_lh_fold)
				y_pred = y_pred + list(y_pred_fold)
				y_test = y_test + list(y_test_fold)

				##################
				# ROC Curve Data #
				##################
				fpr = dict() # False Positive Rates
				tpr = dict() # True Positive Rates for different classifier thresholds
				roc_auc = dict()
				interp_tpr = dict()
				for n in range(len(class_names)):
					if n == 0: 
						fpr[class_names[n]], tpr[class_names[n]], _ = roc_curve(y[test_index], y_score_lh_fold, pos_label = [2]) # negative class, LH
					else:
						fpr[class_names[n]], tpr[class_names[n]], _ = roc_curve(y[test_index], y_score_rh_fold, pos_label = [3]) # positive class, RH
					roc_auc[class_names[n]] = auc(fpr[class_names[n]], tpr[class_names[n]])
					if k == 7:
						print("roc_auc: {0}".format(roc_auc[class_names[n]]))
					interp_tpr = np.interp(mean_fpr, fpr[class_names[n]], tpr[class_names[n]])
					interp_tpr[0] = 0.0
					if n == 0:
						tprs_lh.append(interp_tpr)
					else: 
						tprs_rh.append(interp_tpr)
				aucs.append(roc_auc)

			roc_plot_data = {'aucs': aucs, 'mean_fpr': mean_fpr, 'tprs_lh': tprs_lh, 'tprs_rh': tprs_rh}

			####################################################
			# Metrics for classifier performance, across folds #
			####################################################
			plt.ioff() 										# turn off the interactive interface of matplotlib.pyplot. Otherwise the system might crash
			cmat = confusion_matrix(y_test, y_pred) 		# Confusion Matrix
			tn, fp, fn, tp = cmat.ravel() 					# tn = true negatives (correct LH pred), fp = false positives, fn = false negatives, tp = true positives (correct RH pred)
			pos = tp + fn 									# Total amount of RH class samples in y_test. true positives + false negatives (positives classified as negatives by y_pred)
			neg = tn + fp									# Total amount of LH class samples in y_test. true negatives + false positives (negatives classified as positives by y_pred)
			acc = (tp + tn) / len(y_pred) 					# Accuracy (ACC) number of all correct predictions divided by the total number of the dataset
			sn = tp / pos									# Sensitivity (SN) / TRUE Positive Rate (TPR) / Recall (REC) / Accuracy for RH class 
			sp = tn / neg 									# Specificity (SP) / True Negative Rate (TNR) / Accuracy for the LH class
			ppv = tp / (tp  + fp)							# Precision (PREC) of the RH class / Positive Predictive Value (PPV)
			npv = tn / (tn + fn)							# Precisions (PREC) of the LH class / Negative Predictive Value (NPV)
			mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) # Matthews Correlation Coefficient (MCC), measure of quality of binary (two-class) classification

			# Saving all variables
			row[tags[0][i] + tags[1][j] + 'acc'] = acc
			row[tags[0][i] + tags[1][j] + 'sn'] = sn
			row[tags[0][i] + tags[1][j] + 'sp'] = sp
			row[tags[0][i] + tags[1][j] + 'ppv'] = ppv
			row[tags[0][i] + tags[1][j] + 'npv'] = npv
			row[tags[0][i] + tags[1][j] + 'mcc'] = mcc
			row[tags[0][i] + tags[1][j] + 'roc'] = roc_plot_data
			row[tags[0][i] + tags[1][j] + 'cmat'] = cmat
	rows_list.append(row)
classifier_results = pd.DataFrame(rows_list)

#########################################
# Plotting / Analysing Decoding Results #
#########################################
PATH = "/home/rien/Desktop/CAED_figures/" # The path for where to save the results, change this to your any path on your computer. 

# Plotting
plot_violins(classifier_results, PATH)
plot_cmats(classifier_results, PATH)

for subject in subjects:
	plot_cmats(classifier_results, PATH, subject)
	for cross_val in ['kf', 'mc']:
		plot_roc(classifier_results, PATH, subject, cross_val)

# statistical testing
stat_tests(classifier_results)

