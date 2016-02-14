import time
from collections import defaultdict
import plotting
import numpy
numpy.random.seed(13579)

##return the best model choice by name, the change to the list to output alpha,beta
##best = argmax(list)[0]
def argmax(choices):
	choice = max(choices.iterkeys(), key=lambda k: choices[k])
	return choice.split(",")


# optimize alpha using grid search to get maxiumlikelyhood estimate 
def optimize_alpha(modelname,X,y, X_test,y_test, alpha_low=0, alpha_high=2):
	alpha_accuracy =defaultdict(float)
	print "\toptimizing alpha for " + modelname

	# search 10 to the powers of 0 to 2
	for i in range(alpha_low, alpha_high, 1):

		##finer optimizer search
		for factor in range(3,10,3):
			alpha =  10**(i)*factor;
			if (alpha >= 1):
				if modelname == "svm":
						alpha_accuracy[str(alpha)] = scikit_onevsrest(X,y, X_test,y_test,alpha=alpha)
				elif modelname == "knn":
					if (alpha < (X_test.shape[0]/1.5)):
						alpha_accuracy[str(alpha)] = scikit_knn_model(X,y, X_test,y_test,alpha=alpha)
				elif modelname == "dc_tree":
						alpha_accuracy[str(alpha)] = scikit_dc_tree(X,y, X_test,y_test,alpha=alpha)
				elif modelname == "log_reg":
						alpha_accuracy[str(alpha)] = scikit_log_reg(X,y, X_test,y_test,alpha=alpha)
				elif modelname == "ada_boost":
					if int(X_test.shape[0]) > 1000:
						alpha_enlarge = alpha*9
						if alpha_enlarge == 810: ## save time
							alpha_accuracy[str(alpha_enlarge)] = scikit_ada_boost(X,y, X_test,y_test,alpha=alpha_enlarge)
					else: ## when data small dont both with the classifier 
						alpha_enlarge = alpha
						if alpha_enlarge == 90: ## save time
							alpha_accuracy[str(alpha_enlarge)] = scikit_ada_boost(X,y, X_test,y_test,alpha=alpha_enlarge)

				elif modelname == "ran_forest":
						##grid search alph x beta
						for j in range(alpha_low, alpha_high, 1):
							for k in range(5,10,3):
								beta =  10**(j)*k;
								##print "\t\tran_forest alpha: "+str(alpha)+", beta:"+str(beta)
								alpha_accuracy[str(alpha)+","+str(beta)] = scikit_ran_forest(X,y, X_test,y_test,alpha=alpha,beta=beta)
						

	print "\t"+str(sorted(alpha_accuracy.iteritems(), key=lambda x:-x[1])[:3])
	##print alpha_accuracy
	return alpha_accuracy



## evualuate model using cross validation
def select_best_model(training_X,training_Y,data_title=None):
	##print training_X.shape[0]
	##split data into 4 section by every 4th item
	list1x = training_X[0::10]
	list2x = training_X[1::10]
	list3x = training_X[2::10]
	list4x = training_X[3::10]
	list5x = training_X[4::10]	
	list6x = training_X[5::10]
	list7x = training_X[6::10]
	list8x = training_X[7::10]
	list9x = training_X[8::10]
	list10x = training_X[9::10]

	list1y = training_Y[0::10]
	list2y = training_Y[1::10]
	list3y = training_Y[2::10]
	list4y = training_Y[3::10]
	list5y = training_Y[4::10]
	list6y = training_Y[5::10]
	list7y = training_Y[6::10]
	list8y = training_Y[7::10]
	list9y = training_Y[8::10]
	list10y = training_Y[9::10]

	##split = training_X.shape[0] * .90
	seed = 1
	train_set_X = numpy.concatenate((list2x,list3x,list4x,list5x,list6x,list7x))
	train_set_Y = numpy.concatenate((list2y,list3y,list4y,list5y,list6y,list7y))
	validate_set_X = list8x
	validate_set_Y = list8y


	full_train_set_X = numpy.concatenate((list2x,list3x,list4x,list5x,list6x,list7x,list8x))
	full_train_set_Y = numpy.concatenate((list2y,list3y,list4y,list5y,list6y,list7y,list8y))
	test_set_X = numpy.concatenate((list1x,list9x,list10x))
	test_set_Y = numpy.concatenate((list1y,list9y,list10y))



	##evuate and find best alpha for each model
	start_time = time.clock()
	dict_alpha0 = optimize_alpha("dc_tree",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha0 = int(argmax(dict_alpha0)[0])
	print "\talpha: "+str(best_alpha0)
	train_time0 = time.clock() - start_time

	start_time = time.clock()
	y_test0 = scikit_dc_tree(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha = best_alpha0)
	predict0 = time.clock() - start_time
	print

	start_time = time.clock()
	dict_alpha1 = optimize_alpha("knn",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha1 = int(argmax(dict_alpha1)[0])
	print "\talpha: "+str(best_alpha1)
	train_time1 = time.clock() - start_time

	start_time = time.clock()
	y_test1 = scikit_knn_model(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha = best_alpha1)
	predict1 = time.clock() - start_time
	print

	start_time = time.clock()
	dict_alpha2 = optimize_alpha("log_reg",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha2 = float(argmax(dict_alpha2)[0])
	print "\talpha: "+str(best_alpha2)
	train_time2 = time.clock() - start_time

	start_time = time.clock()
	y_test2 = scikit_log_reg(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha = best_alpha2)
	predict2 = time.clock() - start_time
	print


	##ensemble classifiers:

	start_time = time.clock()
	dict_alpha3 = optimize_alpha("ran_forest",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha3 = int(argmax(dict_alpha3)[0])
	best_beta3 = int(argmax(dict_alpha3)[1])
	print "\talpha: "+str(best_alpha3)+" beta: "+str(best_beta3)
	train_time3 = time.clock() - start_time

	start_time = time.clock()
	y_test3 = scikit_ran_forest(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha=best_alpha3,beta=best_beta3)
	predict3 = time.clock() - start_time
	print

	start_time = time.clock()
	dict_alpha4 = optimize_alpha("ada_boost",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha4 = int(argmax(dict_alpha4)[0])
	print "\talpha: "+str(best_alpha4)
	train_time4 = time.clock() - start_time

	start_time = time.clock()
	y_test4 = scikit_ada_boost(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha = best_alpha4)
	predict4 = time.clock() - start_time
	print



	""" SVM extras too slow
	start_time = time.clock()
	dict_alpha2 = optimize_alpha("svm",train_set_X,train_set_Y,validate_set_X,validate_set_Y)
	best_alpha2 = float(argmax(dict_alpha2))
	print "\t"+str(best_alpha2)
	train_time2 = time.clock() - start_time

	start_time = time.clock()
	y_test2 = scikit_onevsrest(full_train_set_X, full_train_set_Y, test_set_X,test_set_Y,alpha = best_alpha0)
	predict2 = time.clock() - start_time
	print
	"""


	## consolidate and output final result
	dict_choices = {"dc_tree":y_test0,"knn":y_test1,"log_reg":y_test2,"ran_forest":y_test3,"ada_boost":y_test4}
	dict_alpha = {"dc_tree":best_alpha0,"knn":best_alpha1,"log_reg":best_alpha2,"ran_forest":best_alpha3,"ada_boost":best_alpha4}
	dict_beta = {"ran_forest":best_beta3}
	print dict_choices
	final_model = argmax(dict_choices)[0]


	##prep for bar graphs
	dict_errors = dict_choices
	for key, value in dict_errors.iteritems():
		dict_errors[key] = 1- value 
	errors = [dict_errors["dc_tree"],dict_errors["knn"],dict_errors["log_reg"]]
	traintimes = [train_time0,train_time1,train_time2]
	predicttimes = [predict0,predict1,predict2]

	##plot accuracy
	plotting.bar_graph(data_title,"decisiontree","knn","log_reg",[1,2,3],errors,maxy=max(errors)*1.1,ylabel="errors")
	##plot training time
	plotting.bar_graph(data_title,"decisiontree","knn","log_reg",[1,2,3],traintimes,maxy=max(traintimes)*1.1,ylabel="training times(s)")
	##plot prediction time
	plotting.bar_graph(data_title,"decisiontree","knn","log_reg",[1,2,3],predicttimes,maxy=max(predicttimes)*1.1,ylabel="prediction times(s)")



	#prep for line graphs
	alpha_X =[]
	alpha_X_ints = []
	for i in range(0, 2, 1):
		for factor in range(3,10,3):
			alpha =  10**(i)*factor;
			if (alpha > 1):
				alpha_X.append(str(alpha))			
				alpha_X_ints.append(alpha)

	alpha_Y_dc = []
	alpha_Y_knn = []
	alpha_Y_log_reg = []
	dict_error0 = dict_alpha0
	for key, value in dict_alpha0.iteritems():
		dict_error0[key] = 1.0- value 

	dict_error1 = dict_alpha1
	for key, value in dict_alpha1.iteritems():
		dict_error1[key] = 1.0- value 

	dict_error2 = dict_alpha2
	for key, value in dict_alpha2.iteritems():
		dict_error2[key] = 1.0- value 


	for xvalue in alpha_X:
		alpha_Y_dc.append(dict_error0[xvalue])
		alpha_Y_knn.append(dict_error1[xvalue])
		alpha_Y_log_reg.append(dict_error2[xvalue])

	alpha_Y = alpha_Y_dc,alpha_Y_knn,alpha_Y_log_reg


	#plot line graph
	plotting.line_graph_alpha_error(data_title,"dc_tree","knn","log_reg",alpha_X_ints,alpha_Y)

	return final_model,dict_alpha[final_model],best_beta3



def run_model(modelname,X,y, X_test,alpha=None,beta=None):
	best_alpha=alpha
	prediction = None

	if modelname == "svm":
		prediction = scikit_onevsrest(X,y,X_test,alpha = best_alpha)
	elif modelname == "knn":
		prediction = scikit_knn_model(X,y,X_test,alpha = best_alpha)
	elif modelname == "dc_tree":
		prediction = scikit_dc_tree(X,y,X_test,alpha = best_alpha)
	elif modelname == "log_reg":
		prediction = scikit_dc_tree(X,y,X_test,alpha = best_alpha)
	elif modelname == "ada_boost":
		prediction = scikit_ada_boost(X,y,X_test,alpha = best_alpha)
	elif modelname == "ran_forest":
		prediction = scikit_ran_forest(X,y,X_test,alpha = best_alpha,beta=beta)

	return prediction


###################################################################################################
#models


def scikit_dc_tree(X,y, X_test,y_test=None,alpha=None):
    from sklearn import tree
    predictions = tree.DecisionTreeClassifier(random_state=0,max_depth=alpha,criterion='gini').fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions


def scikit_knn_model(X,y, X_test,y_test=None,alpha=None):
	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=alpha)
	neigh.fit(X, y) 
	y_test_predict =neigh.predict(X_test)


	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(y_test_predict):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy = float(correctcount)/totalcount
	    return accuracy

	return y_test_predict

def scikit_log_reg(X,y, X_test,y_test=None,alpha=None):
    from sklearn import linear_model
    predictions = linear_model.LogisticRegression(C=alpha, penalty='l2').fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions



def scikit_onevsrest(X,y, X_test,y_test=None,alpha=None):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    predictions = OneVsRestClassifier(LinearSVC(random_state=0,C=alpha)).fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions


def scikit_ran_forest(X,y, X_test,y_test=None,alpha=None,beta =None):
	from sklearn.ensemble import RandomForestClassifier
	predictions = RandomForestClassifier(n_estimators=alpha,max_depth=beta,random_state=0,n_jobs=-1).fit(X, y).predict(X_test)

	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

	return predictions


def scikit_ada_boost(X,y, X_test,y_test=None,alpha=None,beta =None):
	from sklearn.ensemble import AdaBoostClassifier
	predictions = AdaBoostClassifier(n_estimators=alpha,random_state=0).fit(X, y).predict(X_test)

	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

	return predictions
