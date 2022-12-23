clear;close all;

%% path
listname = 'test_Inria_SLFD.txt'; % data list path
folder = 'Y:\Dataset_Inria_synthetic\SLFD'; % data path
flow_dir = '.\flow_sources\test_flow_RAFT_Inria_SLFD_7x5x2.mat';
savepath = 'test_ALFR_Inria_SLFD_0-20_5x5_Y_flow_PWCNet_warpedFlow.mat';
an = 5;

%% initilization
lf = [];
LF_name = {};
count = 0;

%% read datasets
% load lf list
f = fopen(listname);
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f);

% read lfs
for i_lf = 1:length(list)
    lfname = list{i_lf};
    read_path = fullfile(folder,lfname);
    lf_gt_rgb = read_inria(read_path,9,an);%[x,y,c,u,v]
    lf_gt_ycbcr = rgb2ycbcr_5d(lf_gt_rgb);
    lf_gt_y = squeeze(lf_gt_ycbcr(:,:,1,:,:)); %[x,y,u,v]
    lf = cat(5,lf,lf_gt_y);%[x,y,u,v,N]
    LF_name = cat(1,LF_name,abs(lfname));
end

lf = permute(lf,[5,3,4,1,2]);  %[x,y,u,v,N]==>[N,u,v,x,y]

%% forward warping flows 
load(flow_dir);
[num,u,a_sparse,h,w] = size(flow);
[X, Y] = meshgrid(1:w, 1:h); %integer coordinate
flow_warped = zeros( num, h, w, 2, an, an, 'single');
for i=1:num
    for iy=1:an
        for ix=1:an
            item = 0;
            for j=1:an-1:an
                item = item + 1;
                curY = Y;
                curX = X + abs(ix-j)*double(squeeze(flow(i,iy,item,:,:))/(an-1));
                F = scatteredInterpolant(reshape(curX,[],1),reshape(curY,[],1),reshape(double(flow(i,iy,item,:,:)/(an-1)),[],1));
                flow_warped(i, :, :, item, iy, ix) = F(X,Y);
            end
        end
    end
    fprintf('lf %d \n', i);
end
flow_warped = permute(flow_warped,[1,5,6,2,3,4]); %[N,x,y,2,u,v] -> [N,u,v,x,y,2]

%% save data
if exist(savepath,'file')
    fprintf('Warning: replacing existing file %s \n', savepath);
    delete(savepath);
end
save(savepath, 'lf', 'flow', 'flow_warped', 'LF_name','-v6');

%% display
figure,imshow(squeeze(lf(3,3,3,:,:)));
figure,imshow(squeeze(flow(3,3,1,:,:))), colormap(jet), caxis([-80,80]);
figure,imshow(squeeze(flow(3,3,2,:,:))), colormap(jet), caxis([-80,80]);
figure,imshow(squeeze(flow_warped(3,3,3,:,:,1))), colormap(jet), caxis([-60,60]);
figure,imshow(squeeze(flow_warped(3,3,3,:,:,2))), colormap(jet), caxis([-60,60]);

%% functions

function lf_ycbcr = rgb2ycbcr_5d(lf_rgb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lf_rgb [h,w,3,ah,aw] --> lf_ycbcr [h,w,3,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if length(size(lf_rgb))<5
    error('input must have 5 dimensions');
else
    lf_ycbcr = zeros(size(lf_rgb),'like',lf_rgb);
    for v = 1:size(lf_rgb,4)
        for u = 1:size(lf_rgb,5)
            lf_ycbcr(:,:,:,v,u) = rgb2ycbcr(lf_rgb(:,:,:,v,u));
        end
    end

end
end


function lf = read_inria(read_path, an_org, an_new)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read [h,w,3,ah,aw] data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = 512;
W = 512;
lf = zeros(H,W,3,an_org,an_org,'uint8');
for v = 1:an_org
    for u = 1:an_org
        imgname = sprintf('lf_%d_%d.png',v,u);      
        sub = imread(fullfile(read_path,imgname));
        lf(:,:,:,v,u) = sub;
    end
end
an_crop = ceil((an_org - an_new) / 2 );
lf = lf(:,:,:,1+an_crop:an_new+an_crop,1+an_crop:an_new+an_crop); %[h,w,c,ah,aw]
end
