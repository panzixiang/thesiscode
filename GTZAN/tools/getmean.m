allmeanmfcc = zeros(32,1000);
for i = 1:length(allmfcc)
    songmfcc = cell2mat(allmfcc(i));
    meanmfcc = mean(songmfcc,2);
    allmeanmfcc(:,i) = meanmfcc;   
end
blue = allmeanmfcc(1:32, 1:100);
classical = allmeanmfcc(1:32, 101:200);
country = allmeanmfcc(1:32, 201:300);
disco = allmeanmfcc(1:32, 301:400);
hiphop = allmeanmfcc(1:32, 401:500);
jazz = allmeanmfcc(1:32, 501:600);
metal = allmeanmfcc(1:32, 601:700);
pop = allmeanmfcc(1:32, 701:800);
reggae = allmeanmfcc(1:32, 801:900);
rock = allmeanmfcc(1:32, 901:1000);

figure

subplot(2,5,1)
hist(blue)
title('Blues')


subplot(2,5,2)
hist(classical)
title('Classical')


subplot(2,5,3)
hist(country)
title('Country')


subplot(2,5,4)
hist(disco)
title('Disco')


subplot(2,5,5)
hist(hiphop)
title('HipHop')


subplot(2,5,6)
hist(jazz)
title('Jazz')


subplot(2,5,7)
hist(metal)
title('Metal')


subplot(2,5,8)
hist(pop)
title('Pop')


subplot(2,5,9)
hist(reggae)
title('Reggae')


subplot(2,5,10)
hist(rock)
title('Rock')

