% function Mat=sparseCoding(image_Origin,patchSize)
clear all;clc
image_Origin = imread('jq1_meanshift.jpg.jpg');
patchSize =11;

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
for i = fix(1.5*patch_Weight)+1:size(image_Gray,2)-fix(1.5*patch_Weight)                 %图块的左上角x fix为向小取整
    disp(i)
    for j = fix(1.5*patch_Height)+1:size(image_Gray,1)-fix(1.5*patch_Height)             %图块的左上角y
        patch = image_Gray(j-fix(patch_Height/2):j+fix(patch_Height/2),i-fix(patch_Weight/2):i+fix(patch_Weight/2));  %中心图块
        patch_Vector = patch';              % 实现patch2Vector功能
        patch_Vector = patch_Vector(:);

        %获取斑块的环绕区块
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
        %获取斑块的环绕区块
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

% 归一化后绘图
nMat = rgb2gray(Mat);
Max = max(max(nMat));
Scale = 255/Max;
nMat = nMat*Scale;
imshow(nMat)
colormap(jet)
save resultPatchSize15
