clear; clc; close; 

renderer = 'ENTER YOUR RENDERER HERE';
pythonProgram = 'ENTER YOUR PYTHON PROGRAM HERE';
for n = 1 : 20

root = ['Example', num2str(n)] ;


normalName = fullfile(root, 'normal.png');
roughnessName = fullfile(root, 'rough.png');
diffuseName = fullfile(root, 'albedo.png');
envName = fullfile(root, 'env.npz');
meshNewName = 'bunny.ply';
meshNewNameParts = strsplit(meshNewName, '.');
meshInfoName = strrep(meshNewName, ['.', meshNewNameParts{end}], 'Init.mat');
load(meshInfoName );

for insertionId = 0 : 5

originName = fullfile(root, 'im');
    
for hehe = 0 : insertionId-1
    originName = [originName, '_new'];
end
originName = [originName, '.png'];

cropInfo = strrep(originName, '.png', '_new.mat');
if ~exist(cropInfo, 'file')
    break;
end
load(cropInfo);

suffix = originName(end-3:end);
fovX = 63.4149;
mode = 'RUN';
isGama = false;
r = 0.8; g =0.8; b = 0.8;
roughness = 0.2;
isNewMask = true;

%% Read Images
normal = imread(normalName );
height = size(normal, 1);
width = size(normal, 2);

[xgrid, ygrid] = meshgrid(1:width, 1:height);
mask = ones(height, width);

%% Generate Mask
im = imread(originName );
im = imresize(im, [height, width] );

% Location of the plance needed to be saved.
x = xMask; y = yMask;
scale = scaleSphere;
for n = 1:4
    if n == 4
        seg = [x(1)-x(n), y(1) - y(n)];
    else
        seg = [x(n+1) - x(n), y(n+1) - y(n)];
    end
    mask =  mask .* ( ( (xgrid - x(n)) * seg(2) ...
        - (ygrid - y(n)) * seg(1) ) > 0) ;
end


%% Generate 3D point
normalX = normal(:, :, 1);
normalY = normal(:, :, 2);
normalZ = normal(:, :, 3);
normalX = mean(single(normalX(imerode(mask == 1, strel('disk', 1) ) ) ) );
normalY = mean(single(normalY(imerode(mask == 1, strel('disk', 1) ) ) ) );
normalZ = mean(single(normalZ(imerode(mask == 1, strel('disk', 1) ) ) ) );
vn = [normalX; normalY; normalZ ];
vn = vn ./ 127.5 - 1;
vn = vn ./ sqrt( sum(vn .* vn) );
vn(1) = -vn(1); vn(3) = -vn(3);
if exist('vnSelected', 'var')
    vn = vnSelected;
end

atanFovX = tand(fovX / 2.0);
atanFovY = atanFovX / width * height;
vx = (( (width + 1) / 2.0) - x ) ./ ( (width -1) / 2.0) .* atanFovX;
vy = (( (height + 1) / 2.0) - y ) ./ ( (height -1) / 2.0 ) .* atanFovY;
v = [vx, vy, ones(4, 1)];

v = [vx, vy, ones(4, 1)];
for n = 2:4
    d = (v(1, :) * vn) / (v(n, :) * vn);
    assert(d > 0);
    v(n, :) = d * v(n, :);
end

vtx = (x - 1) / single(width -1);
vty = (height - y) / single(height -1);
vt = [vtx, vty];

%% Output 3D mesh 
meshName = strrep(normalName, 'normal', 'mesh');
meshName = strrep(meshName, 'png', 'obj');
fid = fopen(meshName, 'w');
for n = 1 : 4
    fprintf(fid, 'v %.5f %.5f %.5f\n', v(n, 1), v(n, 2), v(n, 3) );
end
for n = 1 : 4
    fprintf(fid, 'vt %.5f %.5f\n', vt(n, 1), vt(n, 2) );
end
fprintf(fid, 'vn %.5f %.5f %.5f\n', vn(1), vn(2), vn(3) );
fprintf(fid, 'f 1/1/1 2/2/1 3/3/1\n' );
fprintf(fid, 'f 1/1/1 3/3/1 4/4/1\n' );

%% Pick up the point to place the object
xobj = xObj; yobj = yObj;
vimg = [single(xobj - 1) / single(width-1), single(yobj-1) / single(height-1) ];

vxobj = (( (width + 1) / 2.0) - xobj ) ./ ( (width -1) / 2.0) .* atanFovX;
vyobj = (( (height + 1) / 2.0) - yobj ) ./ ( (height -1) / 2.0 ) .* atanFovY;
vobj = [vxobj, vyobj, 1];
if strcmp(mode, 'RUN')
    d = (v(1, :) * vn) / (vobj * vn);
elseif strcmp(mode, 'DEBUG')
    xobj = round(xobj);
    yobj = round(yobj );
    d = depth(yobj, xobj);
end
assert(d > 0)
vobj = vobj * d;

Dist = [norm(vobj-v(1, :) ), norm(vobj-v(2, :) )];
scale = scale * min(Dist );

infoName = strrep(normalName, 'normal', 'info');
infoName = strrep(infoName, 'png', 'mat');
save(infoName, 'vn', 'vobj', 'scale', 'vimg');

%% Gamma correct diffuse albedo
if isGama
    diffuse = single(imread(diffuseName ) ) / 255.0;
    diffuse = diffuse .^ (1.0/2.2);
    diffuse = uint8(255 * diffuse );
    diffuseName = strrep(diffuseName, '.png', '_gamma.png');
    imwrite(diffuse, diffuseName );
end


%% Generate the xml file

cmd = sprintf( ['%s generateXML.py --envName %s --roughnessName %s ' ...
    ,'--meshName %s --meshNewName %s --infoName %s --diffuseName %s --rColor %.3f ' ...
    ,'--gColor %.3f --bColor %.3f --roughness %.3f '...
    , '--meshTranslate %.3f %.3f %.3f --meshRotateAxis %.3f %.3f %.3f ' ...
    , '--meshRotateAngle %.3f --meshScale %.3f'], ...
    pythonProgram, envName, roughnessName, meshName, meshNewName, infoName, diffuseName,  ...
    r, g, b, roughness, meshTranslate(1), meshTranslate(2), meshTranslate(3), ...
    meshRotateAxis(1), meshRotateAxis(2), meshRotateAxis(3), meshRotateAngle, meshScale );
fprintf([cmd, '\n'] );
system(cmd );

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Compute our method %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Render the image and get the difference
envMatName = strrep(envName, '.npz', '.mat');
env = load(envMatName ); env = env.env;
envOrigName = strrep(envMatName, '.mat', 'Origin.mat' );
envOrig = load(envOrigName ); envOrig = envOrig.env;
env = flip(env, 3);
envOrig = flip(envOrig, 3 );
hdrwrite(max(env, 0), strrep(envMatName, 'mat', 'hdr') );

%% Run the rendering 
xmlFile1 = 'scene.xml';
xmlFile2 = strrep(xmlFile1, '.xml', '_obj.xml');
xmlFile3 = strrep(xmlFile1, '.xml', '_bkg.xml');
output1 = strrep(xmlFile1, '.xml', '.rgbe');
output2 = strrep(xmlFile2, '.xml', '.rgbe');
output3 = strrep(xmlFile3, '.xml', '.rgbe');

cmd1 = sprintf('%s -f %s -o %s -m %d', renderer, xmlFile1, output1, 0);
cmd3 = sprintf('%s -f %s -o %s -m %d', renderer, xmlFile3, output3, 0);
cmd1_mask = sprintf('%s -f %s -o %s -m %d', renderer, xmlFile1, output1, 4);
cmd2_mask = sprintf('%s -f %s -o %s -m %d', renderer, xmlFile2, output2, 4);
system(cmd1 ); system(cmd3 );
system(cmd1_mask ); system(cmd2_mask );

%% Get the difference and add to the image
hdr1 = hdrread(strrep(output1, '.rgbe', '_1.rgbe') );
hdr2 = hdrread(strrep(output3, '.rgbe', '_1.rgbe') );
mask1 = imread(strrep(output1, '.rgbe', 'mask_1.png') );
mask2 = imread(strrep(output2, '.rgbe', 'mask_1.png') );
mask1 = single(mask1 ) / 255.0;
mask2 = single(mask2 ) / 255.0;
maskBg = max(mask1 - mask2, 0 );
diff = max(hdr1, 1e-10) ./ max(hdr2, 1e-10) .* maskBg;
diff = min(diff * 1.01, 1);
system(sprintf('rm %s', strrep(output1, '.rgbe', '_1.rgbe') ) );
system(sprintf('rm %s', strrep(output3, '.rgbe', '_1.rgbe') ) );
system(sprintf('rm %s', strrep(output1, '.rgbe', 'mask_1.png') ) );
system(sprintf('rm %s', strrep(output2, '.rgbe', 'mask_1.png') ) );

newName = strrep(originName, suffix, ['_new', suffix]);
ldr = single(imread( originName ) ) / 255.0;
ldr = max(imresize(ldr, [height, width] ), 0);
hdr = ldr .^ (2.2);

mask1 = mask1 == 1;
mask1 = imerode(mask1, strel('disk', 1)); 

hdrNew = hdr .* (1-mask1) + (hdr .* diff) .* mask1;
hdrNew = hdr1 .* mask2 + hdrNew .* (1 - mask2);
hdrNew = max(hdrNew, 0);
hdrwrite(hdrNew,  newName );
ldr = hdr.^(1.0/2.2);
ldrNew = hdrNew.^(1.0/2.2);
imwrite(ldrNew, newName);


clear xobj yobj x y scale vnSelected

end
end
