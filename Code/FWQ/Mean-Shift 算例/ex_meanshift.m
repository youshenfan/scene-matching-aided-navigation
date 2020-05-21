clear all; close all; clc;
%% mean shift
% 均值漂移路径
% simplest mean shift example
% find centroid of 300 points
%test data
mu=[0 0];  %  (x和y坐标的中心值)
S=[30 0;0 35];  %coff 协方差
data=mvnrnd(mu,S,300);   %generate test data cov(data) similar to S 产生300个高斯分布数据点
plot(data(:,1),data(:,2),'o');
% para set
h=5;    %kernel   核的大小
x=[data(1,1) data(1,2)];    %iter init value  随机选中一个初始点
pre_x=[0 0];

% iteration
hold on
res = [];   % 记录差值
while norm(pre_x-x)>0.0001  % 收敛判据（pre_x指前一中心值，x指当前中心值）
    res = [res norm(pre_x-x)];    %（加入的向量 norm(pre_x-x)的欧几里德范数）
    pre_x=x;                         % 更新中心值
    plot(x(1),x(2),'r+');            % 绘制新中心值
    u=0;        % nuerator sum   初始化
    d=0;        %denominator sum 初始化
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
% ————————————————
% 版权声明：本文为CSDN博主「Strangers_bye」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
% 原文链接：https://blog.csdn.net/u012526003/java/article/details/51206551