function n_saved = openvein_matlab_extract_one(inDir, outDir, tag)
%OPENVEIN_MATLAB_EXTRACT_ONE Run one OpenVein extractor on a folder of PNGs.
%
%   n_saved = openvein_matlab_extract_one(inDir, outDir, tag)
%
%   inDir  - folder with input PNG images (sorted by name for filename mapping)
%   outDir - folder for output feature PNGs (same basenames as inputs)
%   tag    - extractor tag: RLT, MC, WLD, PC, GF, or EMC
%
%   Requires OpenVein toolkit on the MATLAB path (addpath before calling).
%   Called from Python via MATLAB Engine — not intended for manual use.

    featType = local_tag_to_feature_type(tag);

    files = dir(fullfile(inDir, '*.png'));
    files = files(~[files.isdir]);
    [~, idx] = sort({files.name});
    files = files(idx);

    if isempty(files)
        error('openvein:noImages', 'No PNG images in: %s', inDir);
    end

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    m = Matcher(1, 1, 1);
    m.VerboseOutput = 'false';
    m = m.readImages(inDir, '*.png', 'DataBaseType', 'CUSTOMFOLDER');

    ppMethods = {'ToDouble', 'LeeRegion', 'Zhao09', 'CLAHE', 'Zhang09'};
    ppArgs = repmat({{}}, 1, numel(ppMethods));
    m = m.preprocessImageSet(ppMethods, ppArgs);

    m.featureExtractionType = featType;
    m = m.calculateFeatureSet();

    nrSubjects = size(m.FeatureSet.Images, 1);
    nrFingers = size(m.FeatureSet.Images, 2);
    nrImages = size(m.FeatureSet.Images, 3);

    k = 0;
    for s = 1:nrSubjects
        for f = 1:nrFingers
            for i = 1:nrImages
                feat = m.FeatureSet.Images{s, f, i};
                if isempty(feat)
                    continue
                end

                k = k + 1;
                if k > numel(files)
                    error('openvein:filenameMapping', ...
                        'Filename mapping failed: more features than files (k=%d).', k);
                end

                [~, baseName, ~] = fileparts(files(k).name);
                feat8 = uint8(feat) * 255;
                imwrite(feat8, fullfile(outDir, [baseName, '.png']));
            end
        end
    end

    n_saved = k;
end


function featType = local_tag_to_feature_type(tag)
    switch upper(strtrim(char(tag)))
        case 'RLT'
            featType = FeatureType.RepeatedLineTracking;
        case 'MC'
            featType = FeatureType.MaximumCurvature;
        case 'WLD'
            featType = FeatureType.HuangWideLine;
        case 'PC'
            featType = FeatureType.PrincipalCurvature;
        case 'GF'
            featType = FeatureType.KumarGabor;
        case 'EMC'
            featType = FeatureType.EMC;
        otherwise
            error('openvein:unknownTag', 'Unknown extractor tag: %s', tag);
    end
end
