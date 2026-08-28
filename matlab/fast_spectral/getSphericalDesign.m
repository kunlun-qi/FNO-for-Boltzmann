function [sph_tmp] = getSphericalDesign(n)
% symmetric spherical design

N = n;

if N == 6
    fileID = fopen('ss003.00006.txt','r');
elseif N == 12
    fileID = fopen('ss005.00012.txt','r');
elseif N == 32
    fileID = fopen('ss007.00032.txt','r');
elseif N == 48
    fileID = fopen('ss009.00048.txt','r');
elseif N == 70
    fileID = fopen('ss011.00070.txt','r');
elseif N == 94
    fileID = fopen('ss013.00094.txt','r');
elseif N == 120
    fileID = fopen('ss015.00120.txt','r');
elseif N == 156
    fileID = fopen('ss017.00156.txt','r');
elseif N == 192
    fileID = fopen('ss019.00192.txt','r');
elseif N == 234
    fileID = fopen('ss021.00234.txt','r');
elseif N == 278
    fileID = fopen('ss023.00278.txt','r');
elseif N == 328
    fileID = fopen('ss025.00328.txt','r');
else
    fprintf('wrong size\n');
end

sizeA = [3 Inf];

A = fscanf(fileID,'%f',sizeA);

fclose(fileID);

sph_tmp.x = A(1,1:n);
sph_tmp.y = A(2,1:n);
sph_tmp.z = A(3,1:n);