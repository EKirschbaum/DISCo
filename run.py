import argparse
import os

from train import Train_DISCoNet
from get_segmentation import get_segmentation
from convert_results import combine_results, convert_single_result

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train DISCo network(s) and \
                                                 predict on neurofinder test data')
    
    parser.add_argument('-p', '--path', default='../neurofinder_videos/', help="path to neurofinder videos (default: %(default)s)")
    
    parser.add_argument('-m','--mode', default='disco', help="decide whether a single network is trained on all videos (disco) or individual networks (discos) (default: %(default)d)")
    
    parser.add_argument('-gpu','--gpu', type=int, default=None, help="ID of the GPU to be used (default: %(default)d)")
    
    parser.add_argument('-a','--add', type=str, default='', help="add ending to outputfile name (default: %(default)d)")
    
    args = parser.parse_args()
    

    
    path = args.path
    
    mode = args.mode
    
    gpu_id = args.gpu
    
    add_ending = args.add    
    
    # create folder for results if not already exists
    if not os.path.exists(path+'results/'):
        os.makedirs(path+'results/')
    
    # create filename for the resultsfile 
    filename = path + 'results/' + mode 
                
    if not add_ending == '':
        filename += '_' + add_ending
    
    # add number if file already exists
    if os.path.exists(filename+'.h5'):
        i = 1
        while os.path.exists(filename+'_'+str(i)+'.h5'):
            i += 1
        filename += '_'+str(i)
    
    if mode == 'discos':
        groups = ['00','01','02','03','04']
        for usedata in groups:
            print(usedata)
            model = Train_DISCoNet(mode=mode,path=path,filename=filename,gpu=gpu_id,usedata=usedata)
            model.run()
            prediction_file = model.filename
            segmentation_file = get_segmentation(prediction_file,mode,usedata)
            print('segmentations saved')
        combine_results(path,segmentation_file)
        print('test results saved')
        
    elif mode == 'disco':
        usedata = None
        model = Train_DISCoNet(mode=mode,path=path,filename=filename,gpu=gpu_id,usedata=usedata) 
        model.run()
        prediction_file = model.filename
        segmentation_file = get_segmentation(prediction_file,mode,usedata)
        print('segmentations saved')
        convert_single_result(path,segmentation_file)    
        print('test results saved')
    print('All done.')
    
    
    
