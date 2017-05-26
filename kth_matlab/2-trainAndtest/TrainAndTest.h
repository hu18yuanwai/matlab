//--------------------------------begin--------------------------------
//-----------------------------by Yikun Lin----------------------------
#ifndef TRAINANDTEST_H_
#define TRAINANDTEST_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include "libsvm/svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static int nClasses = 6; // number of classes
static struct svm_parameter param;	
static struct svm_model *model;
static struct svm_problem train_prob;
static struct svm_node *train_x_space;
static struct svm_node *test_x;
// the candidates of parameter C
static int C[11] = {-5,  -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};
static int cChoice = 11;
//static int C[21] = {-5,  -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//static int cChoice = 21;
// the maximum number of attributes per row
static int max_nr_attr; 

void initiateParam();
bool ReadTrainProblem(FILE* input, int nTrains);
double CrossValidation(int nFolds);
double ReadTestNode(FILE* input, int nTrains);
bool TrainAndTest(int argc, char** argv);

#endif /*TRAINANDTEST_H_*/
//---------------------------------end---------------------------------
