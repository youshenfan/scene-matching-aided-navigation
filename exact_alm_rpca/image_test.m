

clear all;
temp_image = imread('1.jpg');
I = rgb2gray(temp_image);
figure()
subplot(2,3,1)
imshow(I),title('ԭͼ');
subplot(2,3,4)
imhist(I),title('ԭͼֱ��ͼ')%��ʾԭʼͼ��ֱ��ͼ
J = imnoise(I,'salt & pepper',0.02);%��������
subplot(2,3,2)
imshow(J),title('��������');
subplot(2,3,5)
imhist(J),title('����ֱ��ͼ')%��ʾ����ͼ��ֱ��ͼ
G = imnoise(I,'gaussian',0.02,0.02);%��˹����
subplot(2,3,3)
imshow(G);title('��˹����');
subplot(2,3,6)
imhist(G),title('��˹ֱ��ͼ')%��ʾ��˹ͼ��ֱ��ͼ

[A_hat,E_hat, iter] = exact_alm_rpca(double(I));
Img_A=mat2gray(uint8(A_hat));
Img_E=mat2gray(uint8(E_hat));

figure()
subplot(2,3,1)
imshow(Img_A),title('������');   %ֻ��ȥ��С��㣬��������ȫģ��������ԭ�򣺴���ĵ���СͼƬ������ģ����Ч����
                                 %ʵ�飺���ô�ͼ�ֿ鴦��Ȼ��ƴ�ӵķ�ʽ���д���
subplot(2,3,4)
imhist(Img_A),title('ֱ��ͼ')%��ʾԭʼͼ��ֱ��ͼ
subplot(2,3,2)
imshow(Img_E),title('ϡ����');
subplot(2,3,5)
imhist(Img_E),title('ֱ��ͼ')%��ʾ����ͼ��ֱ��ͼ

save("dizhixiang.jpg",Img_A);