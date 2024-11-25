file = open('paths.txt','r')
lines=file.readlines()
file.close()
print(len(lines))
from tqdm import tqdm
import os

file1 = open('report_converged.txt', 'a')
converged = 0
incomplete = []

for line in tqdm(lines):
    if 'I' in line:
         pass
    else:
        try:
            folder=os.listdir(line[:-1])
        except:
            pass 
        else:
            if 'psi4_output3.out' in folder:
                output=open(line[:-1]+'/psi4_output3.out','r')
                output_lines=output.readlines()
                output.close()
                if 'Psi4 exiting successfully' in output_lines[-1]:
                        converged+=1
                        file1.write(line)
                else:
                    incomplete.append(line)
            else:
                incomplete.append(line)
print('converged: ',converged, 'incomplete: ', len(incomplete))
file1.close()

file = open('paths.txt', 'w')
for line in tqdm(incomplete):
     file.write(line)
file.close()
