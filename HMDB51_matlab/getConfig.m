function [videoname, classlabel,fv_dir, vocab_dir, descriptor_path, video_dir,actions, tr_index] = getConfig(split)
    % TODO : Change the paths
    fv_dir = ['/home/hu/remote/Data/HMDB51/feats/feats_split', num2str(split)]; % Path where features will be saved
    vocab_dir = '/home/hu/remote/Data/HMDB51/Vocab';
    descriptor_path = '/home/hu/remote/Data/HMDB51/descriptor';
    video_dir = '/home/hu/remote/hmdb51_org';
    splitdir = '/home/hu/remote/Data/HMDB51/testTrainMulti_7030_splits';
    [videoname, classlabel, tr_index, ~, ~, actions]= getHmdbSplit(split,splitdir);
end
