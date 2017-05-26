//--------------------------------begin--------------------------------
//-----------------------------by Yikun Lin----------------------------
#include "TrainAndTest.h"
void initiateParam()
{
	// initiate param
	param.svm_type = C_SVC;
	param.kernel_type = PRECOMPUTED;
	param.degree = 3;
	param.gamma = 0;	
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1; // a parameter decided by cross-validation
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}


bool ReadTrainProblem(FILE* input, int nTrains)
{
	train_prob.l = nTrains;
	train_prob.y = Malloc(double, train_prob.l);
	train_prob.x = Malloc(struct svm_node *, train_prob.l);
	train_x_space = Malloc(struct svm_node, train_prob.l * (train_prob.l + 2));
	int iEle = 0;
	for (int iRow = 0; iRow < train_prob.l; iRow++)
	{
		train_prob.x[iRow] = &train_x_space[iEle];
		fscanf(input, "%lf", &train_prob.y[iRow]);
		for (int iCol = 0; iCol < train_prob.l + 1; iCol++)
		{
			fscanf(input, "%d:%lf", &train_x_space[iEle].index, &train_x_space[iEle].value);
			iEle++;
		}
		train_x_space[iEle++].index = -1;
	}
	
	for (int iTrain = 0; iTrain < train_prob.l; iTrain++)
	{
		if (train_prob.x[iTrain][0].index != 0)
		{
			return false;
		}
	}
	return true;
}

double CrossValidation(int nFolds)
{
	double best_C = 1;
	int maxCorrect = 0;
	double *target = Malloc(double, train_prob.l);
	
	// for each candidate C, using cross validation
	for (int iChoice = 0; iChoice < cChoice; iChoice++)
	{
		initiateParam();
		param.C = pow(2, C[iChoice]);
		svm_cross_validation(&train_prob, &param, nFolds, target);
		int nCorrect = 0;
		for (int iTrain = 0; iTrain < train_prob.l; iTrain++)
		{
			if (target[iTrain] == train_prob.y[iTrain])
			{
				nCorrect++;
			}
		}
		if (nCorrect > maxCorrect)
		{
			maxCorrect = nCorrect;
			best_C = param.C;
		}
	}
	
	return best_C;
}


double ReadTestNode(FILE* input, int nTrains)
{
	double label;
	max_nr_attr = nTrains + 2;
	test_x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	fscanf(input, "%lf", &label);
	for (int iTrain = 0; iTrain < nTrains + 1; iTrain++)
	{
		fscanf(input, "%d:%lf", &test_x[iTrain].index, &test_x[iTrain].value);
	}
	test_x[nTrains + 1].index = -1;
	if (test_x[0].index != 0)
	{
		return 0;
	}
	return label;
}

bool TrainAndTest(int argc, char** argv)
{
	assert(argc >= 3);
	char* outFile = argv[1];
	int nFolds = argc - 2;
	char** inFiles = new char*[nFolds];
	int** result = new int*[nClasses];
	FILE* pIn;
	FILE* pOut = fopen(outFile, "w");
	int iFold, iClass;
	int nTrains, nTests, iTest;
	int nCorrect = 0;
	int nTotal = 0;
	for (iFold = 0; iFold < nFolds; iFold++)
	{
		inFiles[iFold] = argv[iFold + 2];
	}
	for (iClass = 0; iClass < nClasses; iClass++)
	{
		result[iClass] = new int[nClasses];
		for (int innerClass = 0; innerClass < nClasses; innerClass++)
		{
			result[iClass][innerClass] = 0;
		}
	}
	
	
	// train and test using svm with cross validation
	for (iFold = 0; iFold < nFolds; iFold++)
	{
		pIn = fopen(inFiles[iFold], "r");
		fscanf(pIn, "%d", &nTrains);
		fscanf(pIn, "%d", &nTests);
		
		
		assert(ReadTrainProblem(pIn, nTrains));
		
		double best_C = CrossValidation(5);
		initiateParam();
		param.C = best_C;
		
		model = svm_train(&train_prob, &param);
		
		for (int iTest = 0; iTest < nTests; iTest++)
		{
			double targetLabel;
			assert(targetLabel = ReadTestNode(pIn, nTrains));
			double predictLabel = svm_predict(model, test_x);
			int iTarget = (int)targetLabel - 1;
			int iPredict = (int)predictLabel - 1;
			nTotal++;
			if (iTarget == iPredict)
			{
				nCorrect++;
			}
			result[iTarget][iPredict]++;
			free(test_x);
		}
		fclose(pIn);
		free(train_prob.y);
		free(train_prob.x);
		free(train_x_space);
		svm_free_and_destroy_model(&model);
	}
	float average_accuracy = 0;
	for (iClass = 0; iClass < nClasses; iClass++)
	{
		int nSequences = 0;
		for (int innerClass = 0; innerClass < nClasses; innerClass++)
		{
			fprintf(pOut, "%d\t", result[iClass][innerClass]);
			nSequences += result[iClass][innerClass];
		}
		average_accuracy += ((float)result[iClass][iClass]) / nSequences;
		fprintf(pOut, "\n");
	}
	average_accuracy /= nClasses;
	float accuracy = (float)nCorrect / nTotal;
	fprintf(pOut, "accuracy: %f\n", accuracy);
	fprintf(pOut, "average accuracy: %f\n", average_accuracy);
	fclose(pOut);
	// free all allocated memory
	svm_destroy_param(&param);	
	delete[] inFiles;
	for (iClass = 0; iClass < nClasses; iClass++)
	{
		delete[] result[iClass];
	}
	delete[] result;
	
	return true;
}

//---------------------------------end---------------------------------
