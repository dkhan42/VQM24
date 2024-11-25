from tqdm import tqdm
import os
paths = []
for i in range(1,14):
    filename = 'converged'+str(i)+'.txt'
    fp = open(filename, 'r')
    paths.extend(fp.readlines())
    fp.close()
file = open('scf_todo.txt', 'w')
converged=0
for line in tqdm(paths):
    index=0
    path = line[:-15]
    try:
        dirlist = os.listdir(path[:-2])
    except:
        continue
    if 'scf.out' in dirlist or 'pscf.out' in dirlist:
            file_path = path[:-1] + 'scf.out'
            file2 = open(file_path, 'r')
            output_lines = file2.readlines()
            file2.close()
            if len(output_lines)>1:
                if 'Psi4 exiting successfully' in output_lines[-1]:
                    converged+=1
                    pass
                else:
                     file.write(line[:-1]+'\n')
    else:
        file.write(line[:-1]+'\n')
        #break
file.close()
print('converged: ', converged)

