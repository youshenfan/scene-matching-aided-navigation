function image = fusion(imageCell)
numRow=size(imageCell,1);
numCol=size(imageCell,2);
Weight=size(imageCell{1,1},1);
Height=size(imageCell{1,1},2);
I2=zeros((numRow-1)*Weight,(numCol-1)*Height,size(imageCell,3));
R1=(0:numRow-1)*Weight+1;  %ͼ�������ʼ��
R2=(1:numRow)*Weight;      %ͼ������ֹ��
C1=(0:numCol-1)*Height+1;  %ͼ��������ʼ��
C2=(1:numCol)*Height;      %ͼ��������ֹ��
for i = 1:numRow
    for j = 1:numCol
        I2(R1(i):R2(i),C1(j):C2(j),:) = imageCell{i,j};
    end
end
image=I2;
%image = uint8(I2);
end

