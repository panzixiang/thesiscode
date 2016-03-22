%Change this to the directory containing your data folder

dirn = '/Users/Julia/MachineLearning/HW1';

%intitialize the FV toolbox - you will need to change the filepath appropriately

run('/Users/Julia/vlfeat-0.9.20/toolbox/vl_setup')

%add tools path - you will need to change the filepath appropriately

addpath(genpath('/Users/Julia/MachineLearning/HW1/COS424A1/tools'))

%load all songs into a single struct

[DAT, LB, FNS] = loadAll(dirn);

%extract the MFCC feature

mfcc = cell(1,1000);

for i = 1:length(DAT)

    mfcc{i} = DAT{i}.mfc;

end

%create the structure used as input into the demo_fv

GENDATA.data = mfcc;

GENDATA.class = LB;

GENDATA.classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop',...

    'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};


%run fisher vector

FV = demo_fv(GENDATA, 3, 3)