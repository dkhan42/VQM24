'''
This script calculates all possible, neutral, sum-formulas given the heavy atoms in the valency list. 
Subsequently, graphs (smiles) are generated using the Surge package followed by initial geometries using RDKit  
'''

from itertools import combinations_with_replacement as cr
import os
import numpy as np
import leruli
valency={'C':4,'O':2,'Sx':4,'Sy':6,'N':3,'Nx':5,'F':1,'Si':4,'P':3,'Px':5,'S':2,'Cl':1,'Br':1}
file0=open('failed2.txt','w')
file2=open('submitted2.txt','w')
for i in range(6):
    #List of possible heavy atom combinations
    lis=list(cr(valency,i))

    #Make separate folders for number of heavy atoms
    dir=os.listdir(str(i)+'_heavy_graph')
    
    for tup in lis:
        #Perform integer partitioning through brute-force search
        atoms,counts=np.unique(tup,return_counts=True)
        val=sum([valency[k] for k in tup])
        at=[atoms[k]+str(counts[k]) if counts[k]!=1 else atoms[k] for k in range(len(counts))]
        for j in range((val-2*(i-1))+1):
            mol=''.join(at)+'H'+str(j)

            #Generate graphs using Surge and geometry using RDKit, discard chemical formula if Surge cannot generate graph
            if mol not in dir:
                if os.system('/home/danish/Downloads/surge1_0/surge -u '+mol)==0:
                    #Generate graph using Surge
                    os.system('/home/danish/Downloads/surge1_0/surge -S '+mol+' >'+str(i)+'_heavy_graph/'+mol)
                    file=open(str(i)+'_heavy_graph/'+mol,'r')
                    if len(file.read())==0:
                        os.remove(str(i)+'_heavy_graph/'+mol)
                    else:
                        file.seek(0)
                        k=0
                        for line in file.readlines():
                            #print(line)
                            try:
                                folder=str(i)+'_heavy_geometry/'+mol+'_'+str(k)
                                os.system('mkdir '+folder)
                                geom=mol+'_'+str(k)+'.xyz'
                                file1=open(folder+'/'+geom,'w')

                                #Generate geometry from Graph using RdKit
                                file1.write(leruli.graph_to_geometry(line, "XYZ")['geometry'])
                                file1.close()

                                #Submit Geometry for optimization and conformer search using xTB (Crest)
                                os.system('(cd '+folder+' && leruli task-submit --memory 2000 crest 2.11.2 crest '+geom+')')
                                file2.write(folder)
                                k+=1
                            except:
                                file0.write(line)
                                k+=1
                        file.close()
file0.close()
file2.close()

