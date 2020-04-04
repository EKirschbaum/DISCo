
import h5py as h5
import numpy as np
import json


def combine_results(path,filename):    
    groups = ['00','01','02','03','04']
    all_test_results = []
    for group in groups:
        f = h5.File(filename+'.h5','r')
        grp = f[group]
        for dataset in grp.keys():
            if 'neurofinder.'+group in dataset:
                segmentation = grp[dataset][...]
                
                seg = segmentation[0]
                
                unique = np.unique(seg)
    
                seg_dict = []
                for u in unique:
                    if u==0:
                        continue
                    all_x, all_y = np.where(seg==u)
                    coords_list = []
                    for x,y in zip(all_x,all_y):
                        coords_list.append([int(x),int(y)])
                    # remove tiny segments
                    if len(coords_list) < 25:
                        continue
                    seg_dict.append({"coordinates" : coords_list})
                if len(seg_dict) < 1:
                    continue
                    
                if 'test' in dataset:
                    all_test_results.append({'dataset' : dataset[len('neurofinder.'):],
                                                'regions' : seg_dict})
                
        f.close()
    save_file = path+filename+'_test_results.json'
    with open(save_file,'w') as out_file:
        json.dump(all_test_results,out_file)
            

def convert_single_result(path,filename):    
    all_test_results = []
    f = h5.File(filename+'.h5','r')
    for dataset in f.keys():
        segmentation = f[dataset][...]
        
        seg = segmentation[0]
        
        unique = np.unique(seg)

        seg_dict = []
        for u in unique:
            if u==0:
                continue
            all_x, all_y = np.where(seg==u)
            coords_list = []
            for x,y in zip(all_x,all_y):
                coords_list.append([int(x),int(y)])
            # remove tiny segments
            if len(coords_list) < 25:
                continue
            seg_dict.append({"coordinates" : coords_list})
        if len(seg_dict) < 1:
            continue
            
        if 'test' in dataset:
            all_test_results.append({'dataset' : dataset[len('neurofinder.'):],
                                        'regions' : seg_dict})
        
    f.close()
    save_file = path+filename+'_test_results.json'
    with open(save_file,'w') as out_file:
        json.dump(all_test_results,out_file)
            
