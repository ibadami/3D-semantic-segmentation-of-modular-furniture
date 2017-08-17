%Author: Manu Tom
%Date: 18-May-2016
% Depth and orientation angle dependent axial kinect noise
% orientation angle dependent lateral kinect noise

close all;
clear all;


%%%INPUTS%%
NOISE_TYP = 'both';% 'axial', 'both', 'lateral'
BLUR ='yes';%'no', 'yes'
angleX = 10;% in degrees
angleY = 20;% in degrees
angleZ = 40;% in degrees

angleEff = max(angleX,angleY) % angle around Z axis not at all a factor

thetaRadAbs = abs(angleEff)*pi/180;

files = dir('*.png');
outputPath = [pwd,'/Noisy'];
mkdir(outputPath);
files = files';

%Axial Noise parameters
sigAxial = 0.0012; 

% Lateral noise parameters
sigLx = 0.8 + 0.035*thetaRadAbs/(pi/2 -thetaRadAbs);
sigLy = 0.8 + 0.035*thetaRadAbs/(pi/2 -thetaRadAbs);
muLx = 2;
muLy = 2;

for file = 1 : size(files,2)
    img = imread(files(file).name);
    depth = double(img(:,:,1))/double(255)*double(2.8);    
    
    lateralAxialNoisyImg = uint8(zeros(size(depth)));
    sig_Axial = zeros(size(depth));
    
    for h =1: size(depth,1)
        for w =1:size(depth,2)
            if(abs(angleY)>=10 && abs(angleY)<=60)
                sig_Axial(h,w)  = sigAxial + 0.0019*(depth(h,w)-0.4)*(depth(h,w)-0.4);      
            elseif(abs(angleY)>60 && abs(angleY)<=90 && depth(h,w) ~= 0)% to prevent shooting to infinity
                sig_Axial(h,w)  = sigAxial + 0.0019*(depth(h,w)-0.4)*(depth(h,w)-0.4) + 0.0001*thetaRadAbs*thetaRadAbs/sqrt(depth(h,w)/(pi/2 -thetaRadAbs)/(pi/2 -thetaRadAbs));      
            else
                sig_Axial(h,w)  = sigAxial;
            end
        end
    end
    depth = uint8(double(depth)/double(2.8)*double(255));
    
    if(strcmp(NOISE_TYP,'lateral'))
        noisyImgAxial = depth;% no axial noise
    elseif (strcmp(NOISE_TYP,'axial') || strcmp(NOISE_TYP,'both'))
        noisyImgAxial = imnoise(depth,'localvar',sig_Axial);% apply axial noise
    end
    
    
    if(strcmp(NOISE_TYP,'axial'))
        lateralAxialNoisyImg =  uint8(noisyImgAxial);% no lateral noise
    elseif (strcmp(NOISE_TYP,'lateral') || strcmp(NOISE_TYP,'both'))
        
        R_x = normrnd(muLx,sigLx,size(depth,1),size(depth,2));
        R_y = normrnd(muLy,sigLy,size(depth,1),size(depth,2));
        
        [X,Y] = meshgrid(1:size(depth,2),1:size(depth,1));
        X_lat = floor(X + R_x);
    
        X_lat(X_lat<1) = 1;
        X_lat(X_lat>size(depth,2)) = size(depth,2);
    
        Y_lat = floor(Y + R_y);
        Y_lat(Y_lat<1) = 1;
        Y_lat(Y_lat>size(depth,1)) = size(depth,1); 
    
        for h =1: size(depth,1)
            for w =1:size(depth,2)
                lateralAxialNoisyImg(h,w) =  uint8(noisyImgAxial(Y_lat(h,w),X_lat(h,w))); % apply lateral noise on axial noised image
            end
        end
        
    end
    
    
    if(strcmp(BLUR,'yes'))
        h = fspecial('gaussian', [5 5], 1.0);
        lateralAxialNoisyImgBlur = imfilter(lateralAxialNoisyImg, h);% gaussian blur the images
    elseif(strcmp(BLUR,'no'))
        lateralAxialNoisyImgBlur = lateralAxialNoisyImg;
    end
    
    imwrite(lateralAxialNoisyImgBlur,[outputPath,'/',files(file).name]);
end

figure(1)
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
%imshow(noisyImgAxial)
imshow(lateralAxialNoisyImgBlur);
    