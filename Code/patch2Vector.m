function [patch_Vector] = patch2Vector(patch)
%��patch�ĻҶȾ���ת��Ϊ������
%patch_Vector = zeors(size(patch,1)*size(patch,2),1);
k = 1;
for i = 1:size(patch,2)   %ˮƽ��������
    for j = 1:size(patch,1)   %��ֱ��������
        patch_Vector(k,1) = patch(i,j);  %��ͼ��һ�����ֿ�
        k = k + 1;
    end
end

end

