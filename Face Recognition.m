X=zeros(120,1);
fmatrix=zeros(120,10404);

for i=1:40
file_name=dir('E:\ahsan\*.png');
im=imread(strcat('E:\ahsan\',file_name(i).name));
I=rgb2gray(imresize(im,[150,150]));
fmatrix(i,:)=extractHOGFeatures(I);
X(i,:)=1;
end

for j=41:80
file_name=dir('E:\arsalan\*.png');
im=imread(strcat('E:\arsalan\',file_name(j-40).name));
I=rgb2gray(imresize(im,[150,150]));
fmatrix(j,:)=extractHOGFeatures(I);
X(j,:)=2;
end

for k=81:120
file_name=dir('E:\ishmal\*.png');
im=imread(strcat('E:\ishmal\',file_name(k-80).name));
I=rgb2gray(imresize(im,[150,150]));
fmatrix(k,:)=extractHOGFeatures(I);
X(k,:)=3;
end

SVMstruct=fitcecoc(fmatrix,X);

I=imread('E:\test1.png');
G=rgb2gray(imresize(I,[150,150]));
querryfeatures=extractHOGFeatures(G);

Z=predict(SVMstruct,querryfeatures);
figure , imshow(I);
if(Z==1)
title('ahsan');
else
    if(Z==2)
        title('arsalan');
else
    if(Z==3)
       title('ishmal');
    else
        title('Person not found');
    end
    end
end



