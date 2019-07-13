function imageCell= segmentation(I,numRow,numCol)
R1=(0:numRow-1)*fix(size(I,1)/numRow)+1;  %ͼ�������ʼ��
R2=(1:numRow)*fix(size(I,1)/numRow);      %ͼ������ֹ��
C1=(0:numCol-1)*fix(size(I,2)/numCol)+1;  %ͼ��������ʼ��
C2=(1:numCol)*fix(size(I,2)/numCol);      %ͼ��������ֹ��
imageCell=cell(numRow,numCol);            %����һ����������*�ݣ���Ԫ��
figure;
%��ͼ��ֿ�
for i = 1:numRow
    for j = 1:numCol
        imageCell{i,j} = I(R1(i):R2(i),C1(j):C2(j),:);
    end
end
end

