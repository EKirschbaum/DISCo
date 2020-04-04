import h5py as h5
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob


from get_affs import GetAffs


class DISCoDataset(Dataset):
    
    def __init__(self, path, device, dtype, mode='disco',usedata=None,  
                 sec_length=100, sec_size=128, n_secs=10):
        """
        Args:
            
        """
        self.dtype = dtype
        self.device = device
        self.sec_length = sec_length
        self.sec_size = sec_size
        self.n_secs = n_secs    
        self.path = path
        
        if usedata is not None:
            training_data_ = []
            self.test_set = []
            training_ = glob.glob(path+'neurofinder.'+usedata+'*.h5')
            test_ = glob.glob(path+'neurofinder.'+usedata+'*.test.h5')
            training_data_ += training_
            self.test_set += test_
        else:
            training_data_ = glob.glob(path+'neurofinder.*.h5')
            self.test_set = glob.glob(path+'neurofinder.*.test.h5')
            
        self.training_data = [item for item in training_data_ if item not in self.test_set]
        del training_data_
        
        self.mode = mode
            
        videos = []
        names = []
        labels = []
        segs = []
        summs = []
        
        for data in self.training_data:
            print('loading ' + data)
            name = data[len(path):-3]
                    
            with h5.File(data,'r') as in_file:
                video = in_file['video'][...].astype(np.int16)
            
            with h5.File(path+'BF_labels.h5','r') as l_file:
                label = l_file[name][...]
            
            with h5.File(path+'gt_segmentations.h5','r') as seg_file:
                seg = seg_file[name][...]
                
            with h5.File(path+'summary_images.h5','r') as s_file:
                summ = s_file[name][...]
            
            names.append(name)
            videos.append(video)
            labels.append(label[None,...])
            segs.append(seg)
            summs.append(summ[None,...])
            
        self.data = {'videos' : videos, 'labels' : labels,  
                     'names' : names, 'summary' : summs, 'segmentations' : segs}
        
        self.data_length = len(names)
        self.predict = False
        
        
        offsets_affs = [[-1, 0], [0, -1], [-5, 0], [0, -5], [-5, -5]]
                
        self.GA = GetAffs(offsets_affs,self.dtype,self.device)
        
        
    def fetch_test_data(self,idx):
        # test set
        videos = []
        names = []
        summs = []
        
        data = self.test_set[idx]
        name = data[len(self.path):-3]
        names.append(name)
            
        with h5.File(data,'r') as in_file:
            video = in_file['video'][...].astype(np.int16)
            videos.append(video)
                    
            summ = np.mean(video,axis=0)
            summs.append(summ[None,...])
        
        
        self.data = {'videos' : videos,  
                          'names' : names, 'summary' : summs}
        
        self.data_length = 1
        
        return()
    
    def fetch_train_data(self,idx):
        # test set
        videos = []
        names = []
        summs = []
        
        data = self.training_data[idx]
        name = data[len(self.path):-3]
        names.append(name)
            
        with h5.File(data,'r') as in_file:
            video = in_file['video'][...].astype(np.int16)
            videos.append(video)
                    
        with h5.File(self.path+'summary_images.h5','r') as s_file:
            summ = s_file[name][...]
            summs.append(summ[None,...])
            
        self.data = {'videos' : videos,  
                          'names' : names, 'summary' : summs}
        
        self.data_length = 1
        
        return()
    
    def get_transform_params(self, seg_o):
        # seg_o: np array, 1 x X x Y
        
        # no transformations for prediction
        hflip = False
        vflip = False
        rots = 0
        maxpool_size = 6
        x_start = 0
        x_stop = seg_o.size(1)
        y_start = 0
        y_stop = seg_o.size(2)
        
        if self.predict == False:
            # random transformations for training
            # vflip
            if np.random.random() > 0.5:
                vflip = True
                    
            # hflip
            if np.random.random() > 0.5:
                hflip = True
                    
            # rotate
            rots = np.random.choice(4,1)[0]
            
            # temporal length for maxpooling
            maxpool_size = np.random.choice(np.arange(3,10),1)[0]
            
            
        
            # random crop
            poss_start = 0
            x_start = np.random.choice(np.arange(poss_start,seg_o.size(1)-self.sec_size),1)[0]
            x_stop = x_start + self.sec_size
            y_start = np.random.choice(seg_o.size(2)-self.sec_size,1)[0]
            y_stop = y_start + self.sec_size
                            
            seg = seg_o[:,x_start:x_stop,y_start:y_stop]
            u = torch.unique(seg)
            while len(u) < 1:
                x_start = np.random.choice(seg_o.size(1)-self.sec_size,1)[0]
                x_stop = x_start + self.sec_size
                y_start = np.random.choice(seg_o.size(2)-self.sec_size,1)[0]
                y_stop = y_start + self.sec_size
                            
                seg = seg_o[:,x_start:x_stop,y_start:y_stop]
                u = torch.unique(seg)
        
        return([hflip,vflip,rots,maxpool_size,x_start,x_stop,y_start,y_stop])
        
    def transform_summlike(self,summ,transform_params,norm=True,comp_affs=False):
        
        hflip,vflip,rots,maxpool_size,x_start,x_stop,y_start,y_stop = transform_params
        
        
        summ = summ[:,x_start:x_stop,y_start:y_stop]
        
        if vflip == True:
            summ = summ.flip(1)
            
        if hflip == True:
            summ = summ.flip(2)
            
        if rots == 1:
            summ = summ.transpose(1,2).flip(1)
            
        if rots == 2:
            summ = summ.flip(1).flip(2)
            
        if rots == 3:
            summ = summ.transpose(1,2).flip(2)
            
        if norm == True:
            s_means = torch.mean(summ)
            s_stds = torch.std(summ)
            summ -= s_means
            summ /= s_stds
        if comp_affs == True:    
            summ = self.GA.get_affs(summ.view(1,1,summ.size(1),summ.size(2)))[:,0,0]
        
        return(summ)
        
    def transform_video(self,video,transform_params, for_eval=False):
            
        hflip,vflip,rots,maxpool_size,x_start,x_stop,y_start,y_stop = transform_params
        
            
        sec_length = int(video.shape[0] / self.n_secs)
        maxpool = nn.MaxPool3d(kernel_size=(maxpool_size,1,1))
            
        
        if self.predict == False:
            sec_starts = np.random.choice(video.shape[0]-sec_length,self.n_secs)
        else:
            sec_starts = np.arange(int(video.shape[0]/sec_length))*sec_length
        
        
        corrs = []
        for n in range(len(sec_starts)):
            video_resized = torch.tensor(video[sec_starts[n]:min(sec_starts[n]+sec_length,video.shape[0]),x_start:x_stop,y_start:y_stop].astype(np.float32),dtype=self.dtype,device=self.device)
            
            video_resized = maxpool(video_resized.reshape(1,1,video_resized.size(0),video_resized.size(1),video_resized.size(2)))[0,0]
            
            if vflip == True:
                video_resized = video_resized.flip(1)
            if hflip == True:
                video_resized = video_resized.flip(2)
            if rots == 1:
                video_resized = video_resized.transpose(1,2).flip(1)
            if rots == 2:
                video_resized = video_resized.flip(1).flip(2)
            if rots == 3:
                video_resized = video_resized.transpose(1,2).flip(2)
                
            corrs.append(self.get_corrs(video_resized))
                
        corrs = torch.stack(corrs)
                
        means = torch.mean(corrs.view(corrs.size(0),corrs.size(1),corrs.size(2)*corrs.size(3)),dim=2).view(corrs.size(0),corrs.size(1),1,1)#,dim=(2,3),keepdim=True)
        stds = torch.std(corrs.view(corrs.size(0),corrs.size(1),corrs.size(2)*corrs.size(3)),dim=2).view(corrs.size(0),corrs.size(1),1,1)#,dim=(2,3),keepdim=True)
            
        corrs -= means
        corrs /= stds
        
        
        return(corrs)
    
    
    def __len__(self):
        return(self.data_length)
        
    def __getitem__(self,idx):
        video = self.data['videos'][idx]
        summary = self.data['summary'][idx]
        name = self.data['names'][idx]
        summ = torch.tensor(summary.astype(np.float32),dtype=self.dtype,device=self.device)
        if self.predict == False:
            seg = self.data['segmentations'][idx]
            label = self.data['labels'][idx]
            # convert to tensors
            label = torch.tensor(label.astype(np.float32),dtype=self.dtype,device=self.device)
            seg = torch.tensor(seg.astype(np.float32),dtype=self.dtype,device=self.device)
            
            transform_params = self.get_transform_params(seg)
        else:
            transform_params = self.get_transform_params(summ)
        
        new_summ = self.transform_summlike(summ, transform_params, norm=True, comp_affs=False)
        corrs = self.transform_video(video, transform_params) 
        
        sample = {'name' : name,
                  'correlations' : corrs, 
                  'summary' : new_summ
                  }
            
        if self.predict == False:
            affs = self.transform_summlike(seg, transform_params, norm=False, comp_affs=True) 
            sample['affinities'] = affs
            
            new_label = self.transform_summlike(label, transform_params, norm=False, comp_affs=False) 
            sample['label'] = new_label
            
                
                  
        return(sample)
    
    

    
    def get_corrs(self,video):
        
        
        offsets = [[1, 0], [0, 1],[1,1], [2, 0], [0, 2], [2,1], [1,2], [2, 2], 
                   [3,0],[0,3], [3,1], [3,2], [1,3], [2,3], [3,3]]  
        
        X = video.size(1)
        Y = video.size(2)
            
        corrs = torch.zeros((len(offsets),X,Y),dtype=self.dtype,device=self.device)
        
        u = video
        u_ = torch.mean(u,dim=0)
        u_u_ = u-u_
        u_u_n = torch.sqrt(torch.sum(u_u_**2,dim=0))
        
        for o,off in enumerate(offsets):
            v = torch.zeros(video.size(),dtype=self.dtype,device=self.device)
            v[:,off[0]:,off[1]:] = video[:,:(video.size(1)-off[0]),:(video.size(2)-off[1])]
            v_ = torch.mean(v,dim=0)
            v_v_ = v-v_
            v_v_n = torch.sqrt(torch.sum(v_v_**2,dim=0))
            
            zaehler = torch.sum(torch.mul(u_u_,v_v_),dim=0)
            nenner = torch.mul(u_u_n, v_v_n)
            
            corrs[o] = torch.where(nenner>0.,zaehler.div(nenner),torch.zeros((X,Y),dtype=self.dtype,device=self.device))
        
        return(corrs)
        

    