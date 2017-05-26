clear;
clc;
% TODO Add paths
addpath('~/lib/vlfeat/toolbox');
vl_setup();
% TODO Add paths
% add open cv to LD_LIB Path
setenv('LD_LIBRARY_PATH','/usr/local/lib/');


% TODO
% add lib linear to path
addpath('~/lib/liblinear/matlab');
% TODO
% add lib svm to path
addpath('~/lib/libsvm/matlab');
% TODO change paths inside getConfig

[fullvideoname, videoname,vocabDir,fv_dir,actionName,descriptor_path,class_category] = getconfig();

st = 1;
send = length(videoname);
fprintf('Start : %d \n',st);
fprintf('End : %d \n',send);
addpath('0-trajectory');
getSalient(st,send,fullvideoname,descriptor_path)

addpath('1-cluster');
samples = 400000;
gmmSize = 256;

gmm =  SelectSalient(gmmSize,samples,fullvideoname,descriptor_path,vocabDir)
encodeVideos(videoname,gmm,fv_dir,descriptor_path,class_category)
