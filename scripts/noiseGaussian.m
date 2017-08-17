
files = dir('*.png');
outputPath = [pwd,'/Noisy'];
mkdir(outputPath);
files = files';
for file = 1 :size(files,2)
    img = imread(files(file).name);
    noisyImg = imnoise(img,'gaussian',0,0.001);
    imwrite(noisyImg,[outputPath,'/',files(file).name]);
end

