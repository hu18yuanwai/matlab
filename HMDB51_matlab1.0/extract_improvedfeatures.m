% TODO: Change the paths and improved trajectory binary paths
function [obj,trj,hog,hof,mbhx,mbhy] = extract_improvedfeatures(videofile)
    [~,nameofvideo,~] = fileparts(videofile);
    outfile = fullfile('~/remote/Data/temp/tmpfiles',sprintf('%s-%1.6f',nameofvideo,tic())); % path of the temporary file
    % Here the path should be corrected
    tempOut = outfile;
    videofile = strrep(videofile, '&', '\&'); outfile = strrep(outfile, '&', '\&');
    videofile = strrep(videofile, '(', '\('); outfile = strrep(outfile, '(', '\(');
    videofile = strrep(videofile, ')', '\)'); outfile = strrep(outfile, ')', '\)');
    videofile = strrep(videofile, ';', '\;'); outfile = strrep(outfile, ';', '\;');
    fprintf('extracting features.\n');

    system(sprintf('%s %s > %s',fullfile('bin','DenseTrackStab'),videofile,outfile));
    
    fprintf('read the temp file.\n');
    data = dlmread(tempOut);

    fprintf('read the temp file ends.\n');
    delete(tempOut);
    fprintf('delete the temp file ends.\n');
    obj = data(:,1:10);
    trj = data(:,11:40);
    hog = data(:,41:41+95);
    hof = data(:,41+96:41+96+107);
    mbhx  = data(:,41+96+108:41+96+108+95);
    mbhy  = data(:,41+96+108+96:41+96+108+96+95);
end
