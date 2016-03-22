allsongcat = [];
for i = 1:length(allmfcc)
    songmfcc = cell2mat(allmfcc(i));
    songcat = [];
    for j = 1:1000
        songcat = vertcat(songcat, songmfcc(:,j));
    end
    allsongcat = horzcat(allsongcat, songcat);   
end
