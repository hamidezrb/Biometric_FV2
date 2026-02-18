# Install 
 ``` pip install numpy opencv-python scikit-image pandas ```

# Matlab


**preprocess Images**
```
clear; clc;

toolkitRoot = 'C:\Users\user\Downloads\OpenVein-Toolkit_v1.0.2';
addpath(genpath(toolkitRoot)); savepath;

m = Matcher(1,1,1);
m.VerboseOutput = 'true';

m = m.readImages('D:\Universität\Biometric Systems\NewProject\iso-iec-29794-9-q1-biometrics\test_images\high_quality', ...
                 '*.png', 'DataBaseType', 'CUSTOMFOLDER');

imshow(m.ImageSet.Images{1,1,1}, []);
title('Original');

ppMethods = {'ToDouble','LeeRegion','Zhao09','CLAHE','Zhang09','Resize'};
ppArgs    = repmat({{}}, 1, numel(ppMethods));
m = m.preprocessImageSet(ppMethods, ppArgs);
imshow(m.ImageSet.Images{1,1,1}, []);
title('After preprocessing');



m.featureExtractionType = FeatureType.RepeatedLineTracking;
m = m.calculateFeatureSet();

disp(size(m.FeatureSet.Images));

feat = m.FeatureSet.Images{1,1,1};
imshow(feat, []);
title('Repeated Line feature');

```

**Save extracted vein images**
```
clear; clc;

toolkitRoot = 'C:\Users\user\Downloads\OpenVein-Toolkit_v1.0.2';
addpath(genpath(toolkitRoot)); savepath;

inDir  = 'D:\Universität\Biometric Systems\NewProject\iso-iec-29794-9-q1-biometrics\test_images\high_quality';
outRoot = 'D:\Universität\Biometric Systems\NewProject\iso-iec-29794-9-q1-biometrics\debug_openvein_features';
if ~exist(outRoot, 'dir'); mkdir(outRoot); end

% ---- Build filename list in a stable order ----
files = dir(fullfile(inDir, '*.png'));
files = files(~[files.isdir]);

% IMPORTANT: sort by name to ensure reproducible mapping
[~, idx] = sort({files.name});
files = files(idx);

% ---- OpenVein read ----
m = Matcher(1,1,1);
m.VerboseOutput = 'true';

m = m.readImages(inDir, '*.png', 'DataBaseType', 'CUSTOMFOLDER');

% ---- Preprocess ----
ppMethods = {'ToDouble','LeeRegion','Zhao09','CLAHE','Zhang09'};
ppArgs    = repmat({{}}, 1, numel(ppMethods));
m = m.preprocessImageSet(ppMethods, ppArgs);


% Try multiple extractors
extractors = {
    FeatureType.RepeatedLineTracking, 'RLT'
    FeatureType.MaximumCurvature,     'MC'
    FeatureType.HuangWideLine,     'WLD'
    FeatureType.PrincipalCurvature,   'PC'
    FeatureType.KumarGabor, 'GF'
    FeatureType.EMC, 'EMC'
};

for e = 1:size(extractors,1)
    featType = extractors{e,1};
    tag      = extractors{e,2};

    outDir = fullfile(outRoot, tag);
    if ~exist(outDir, 'dir'); mkdir(outDir); end



    % ---- Feature extraction ----
    m.featureExtractionType = featType;
    m = m.calculateFeatureSet();
    
    nrSubjects = size(m.FeatureSet.Images, 1);
    nrFingers  = size(m.FeatureSet.Images, 2);
    nrImages   = size(m.FeatureSet.Images, 3);
    
    k = 0;
    for s = 1:nrSubjects
        for f = 1:nrFingers
            for i = 1:nrImages
                feat = m.FeatureSet.Images{s,f,i};
                if isempty(feat); continue; end
    
                k = k + 1;
                if k > numel(files)
                    error("Filename mapping failed: more features than files (k=%d).", k);
                end
    
                % Original filename (no change)
                [~, baseName, ~] = fileparts(files(k).name);
    
                % Save feature image
                feat8 = uint8(feat) * 255; % logical/double -> uint8
                outName = [baseName '.png'];   % <-- EXACT SAME NAME
                % If you want suffix instead, use:
                % outName = [baseName '_repeatedline.png'];
    
                imwrite(feat8, fullfile(outDir, outName));
            end
        end
    end
    
    disp("Saved " + tag + " features to: " + outDir);

end

```
