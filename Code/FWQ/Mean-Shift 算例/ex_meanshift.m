clear all; close all; clc;
%% mean shift
% ��ֵƯ��·��
% simplest mean shift example
% find centroid of 300 points
%test data
mu=[0 0];  %  (x��y���������ֵ)
S=[30 0;0 35];  %coff Э����
data=mvnrnd(mu,S,300);   %generate test data cov(data) similar to S ����300����˹�ֲ����ݵ�
plot(data(:,1),data(:,2),'o');
% para set
h=5;    %kernel   �˵Ĵ�С
x=[data(1,1) data(1,2)];    %iter init value  ���ѡ��һ����ʼ��
pre_x=[0 0];

% iteration
hold on
res = [];   % ��¼��ֵ
while norm(pre_x-x)>0.0001  % �����оݣ�pre_xָǰһ����ֵ��xָ��ǰ����ֵ��
    res = [res norm(pre_x-x)];    %����������� norm(pre_x-x)��ŷ����·�����
    pre_x=x;                         % ��������ֵ
    plot(x(1),x(2),'r+');            % ����������ֵ
    u=0;        % nuerator sum   ��ʼ��
    d=0;        %denominator sum ��ʼ��
    for i=1:300
        %mean shift realization
        k=norm((x-data(i,:))/h).^2;        
        g=(1/sqrt(2*pi))*exp(-0.5*k);
        u=data(i,:)*g+u;
        d=g+d;
    end
    M=u/d;      %position after iteration
    x=M;
end
hold off
figure
plot(res)
% ��������������������������������
% ��Ȩ����������ΪCSDN������Strangers_bye����ԭ�����£���ѭ CC 4.0 BY-SA ��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
% ԭ�����ӣ�https://blog.csdn.net/u012526003/java/article/details/51206551