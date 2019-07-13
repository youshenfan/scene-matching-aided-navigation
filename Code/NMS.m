%threholds ɸѡ��ֵ
%M���ֵ���ƺ����÷�����
%h,n��Ĵ�С
%function  NMS(img_in,threholds,M,w,h)
imgS=imread('223_5_72km_fieldSparse_analysis0413.jpg');
imgL=imread('223_5_72km_fieldLowRank_analysis0412.jpg');
%imgS=imread('223_223_5_8kmSparse.jpg');
%imgL=imread('223_223_5_8kmLowRank.jpg');
%ϡ����͵�����ĵ��ӵ��
img_p=double(imgS).*double(imgL); %matlab������ת��
%���ӽ����������������ֵ
img_p_sort=sort(img_p(:));
max=img_p_sort(length(img_p_sort)); 
%���Ӻ��Ԫ�ع�һ���������ֵΪ��ĸ��
for i=1:size(img_p,1)
    for j=1:size(img_p,2)
        img_p(i,j)=img_p(i,j)/max; 
    end
end
%���Ӻ������ת��ΪͼƬ��ʽ
img_in=uint8(img_p*255);
%�趨��ֵ��ǰM����ֵ����Ŀ�͸�
threholds=0.8*255;
M=5;
w=50;
h=50;
%������ת��Ϊ������
img_col=img_in(:);
%���ͼ�����������߼�����
%img_threholds=img_col>threholds;
%ȡ��������ֵ��Ԫ�ؽ��н�������
Max_img=sort(img_col(img_col>threholds),'descend');  %erro: sort(A(:))������Ԫ������
%ѡ���������к�ǰM������ȵ�Ԫ�أ����ҵ���ԭͼƬ�е�����
V_M=[];
i=1;
V_M=Max_img(1);
for j=2:size(Max_img) 
    if i==M, break; end
    if V_M(i)~=Max_img(j)
        V_M=[V_M;Max_img(j)];%������չ
        i=i+1;
    end
end
[row,col]=find(img_in>=V_M(M));%erro�����������ȷ
s=[];
C=[];
for i=1:size(row)
   ymin=(row(i)-h/2);
   ymax=(row(i)+h/2);
   xmin=(col(i)-w/2);
   xmax=(col(i)+w/2);
   s=[s;img_in(row(i),col(i))];
   if xmin<1
       xmin=1;
   end
   if xmax>size(img_in,2)
       xmax=size(img_in,2);
   end
   if ymin<1
       ymin=1;
   end
   if ymax>size(img_in,1)
       ymax=size(img_in,1);
   end
   C=[C;xmin xmax ymin ymax];  
end
  saved=nms_01(C,s,0.6);
  %img_show=imread('223_223_5_8km.png');
  img_show=imread('223_5_72km_field.png');
  imshow(img_show);
  hold on;
  Corner_box=[];
  for i=1:size(saved)
     x= C(saved(i),:)
     Corner_box=[Corner_box;x];
     rectx = [x(1) x(2) x(2) x(1) x(1)];
     recty = [x(3) x(3) x(4) x(4) x(3)];
     plot(rectx, recty, 'linewidth',2);%����������������
  end
  
  hold off;
       
%end     

