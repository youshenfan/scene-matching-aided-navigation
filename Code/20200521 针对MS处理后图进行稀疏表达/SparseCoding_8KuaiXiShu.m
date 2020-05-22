% function Mat=sparseCoding(image_Origin,patchSize)
clear all;clc
image_Origin = imread('jq1.png');
patchSize =5;

if(numel(size(image_Origin))==3)
    image_Gray = rgb2gray(image_Origin);
else
    image_Gray = image_Origin;
end
patch_Weight = patchSize;
patch_Height = patchSize;

pradH = fix(patch_Height/2);
pradW = fix(patch_Weight/2);
patchL = pradW+1;
patchR = size(image_Gray,2)-pradW;
patchU = pradH+1;
patchB = size(image_Gray,1)-pradH;

    tic
B = cell( size(image_Gray,1)-fix(1.5*patch_Weight),size(image_Gray,2)-fix(1.5*patch_Height));
for i = fix(1.5*patch_Weight)+1:size(image_Gray,2)-fix(1.5*patch_Weight)                 %ͼ������Ͻ�x fixΪ��Сȡ��
    disp(i)
    for j = fix(1.5*patch_Height)+1:size(image_Gray,1)-fix(1.5*patch_Height)             %ͼ������Ͻ�y
        patch = image_Gray(j-fix(patch_Height/2):j+fix(patch_Height/2),i-fix(patch_Weight/2):i+fix(patch_Weight/2));  %����ͼ��
        patch_Vector = patch';              % ʵ��patch2Vector����
        patch_Vector = patch_Vector(:);

        %��ȡ�߿�Ļ�������
        s = zeros(patch_Weight*patch_Height,8);
        tempI = image_Gray((j-patch_Height)-pradH:(j-patch_Height)+pradH,(i-patch_Weight)-pradW:(i-patch_Weight)+pradW)';
        s(:,1) = tempI(:);
        tempI = image_Gray((j-patch_Height)-pradH:(j-patch_Height)+pradH,(i)-pradW:(i)+pradW)';
        s(:,2) = tempI(:);
        tempI = image_Gray((j-patch_Height)-pradH:(j-patch_Height)+pradH,(i+patch_Weight)-pradW:(i+patch_Weight)+pradW)';
        s(:,3) = tempI(:);
        tempI = image_Gray((j)-pradH:(j)+pradH,(i-patch_Weight)-pradW:(i-patch_Weight)+pradW)';
        s(:,4) = tempI(:);
        tempI = image_Gray((j)-pradH:(j)+pradH,(i+patch_Weight)-pradW:(i+patch_Weight)+pradW)';
        s(:,5) = tempI(:);
        tempI = image_Gray((j+patch_Height)-pradH:(j+patch_Height)+pradH,(i-patch_Weight)-pradW:(i-patch_Weight)+pradW)';
        s(:,6) = tempI(:);
        tempI = image_Gray((j+patch_Height)-pradH:(j+patch_Height)+pradH,(i)-pradW:(i)+pradW)';
        s(:,7) = tempI(:);
        tempI = image_Gray((j+patch_Height)-pradH:(j+patch_Height)+pradH,(i+patch_Weight)-pradW:(i+patch_Weight)+pradW)';
        s(:,8) = tempI(:);
        %��ȡ�߿�Ļ�������
        B{j,i}=lasso(double(s),double(patch_Vector),'Lambda',0.2);        
    end
end
    toc

C = zeros(size(image_Origin));
for i =  fix(1.5*patch_Weight+1):size(image_Gray,2)-fix(1.5*patch_Weight)
    for j = fix(1.5*patch_Height)+1:size(image_Gray,1)-fix(1.5*patch_Height)
        temp = B{j,i};
        for k = 1:size(temp,1)
            if(temp(k)~=0)
                C(j,i) = C(j,i) + 1;
            end
        end
    end
end
Mat= uint8(C*255/size(temp,1));

subplot(221)
imshow(image_Origin);
title('ԭͼ')
subplot(222)
imshow(Mat)
title('������ͼ')

% ��һ�����ͼ
nMat = rgb2gray(Mat);
Max = max(max(nMat));
Scale = 255/Max;
nMat = nMat*Scale;
subplot(223)
imshow(nMat)
colormap(jet);
title('������ͼ����һ����')

% ��ֵ����
mask = ones(size(nMat));
mask(nMat<(0.85*max(max(nMat)))) = 0;
mMat = zeros(size(image_Origin));
mMat(:,:,1) = uint8(mask).*image_Origin(:,:,1);
mMat(:,:,2) = uint8(mask).*image_Origin(:,:,2);
mMat(:,:,3) = uint8(mask).*image_Origin(:,:,3);
subplot(224)
% imshow(nMat)

imshow(image_Origin);
hold on
h = imshow(mask);set(h,'AlphaData',0.5)
title('ԭͼ��������ͼ�غ�')


