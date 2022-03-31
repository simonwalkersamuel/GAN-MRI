import shutil
import os
join = os.path.join

Apath = '/home/simon/Desktop/Share/sim64_subsampled'
Bpath = '/home/simon/Desktop/Share/subvol_64'

A,B = False,True
train,test = True,True

cg_dir = '/mnt/ml/cycleGAN/sim2data64'
Atrain_path = join(cg_dir,'trainA')
Btrain_path = join(cg_dir,'trainB')
Atest_path = join(cg_dir,'testA')
Btest_path = join(cg_dir,'testB')

def parse_directory(path):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".nii"):
                res.append(file)
    return res
    
def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            try:
                os.remove(join(folder,filename))
            except Exception as e:
                print('Could not delete {}. Error: {}'.format(filename,e))
    else:
        os.mkdir(folder)    

if A:
    Afiles = parse_directory(Apath)
else:
    Afiles = []
if B:
    Bfiles = parse_directory(Bpath)
else:
    Bfiles = []

print('A: {}, B: {}'.format(len(Afiles),len(Bfiles)))


op = shutil.copy # or move

if train:
    if A:
        clear_folder(Atrain_path)
        for f in Afiles:
            op(join(Apath,f),join(Atrain_path,f))
    if B:
        clear_folder(Btrain_path)
        for f in Bfiles:
            op(join(Bpath,f),join(Btrain_path,f))

if test:
    if A:
        clear_folder(Atest_path)
        for f in Afiles:
            op(join(Apath,f),join(Atest_path,f))
    if B:
        clear_folder(Btest_path)
        for f in Bfiles:
            op(join(Bpath,f),join(Btest_path,f))
