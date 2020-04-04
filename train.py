import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import h5py as h5
import glob

from dataloading import DISCoDataset
from model import DISCoNet
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
        
class Train_DISCoNet(object):
    def __init__(self, path='./',mode='disco', gpu=None, filename='results',
                 usedata=None):
        '''
        Train the network(s) for DISCo
        
        Arguments:
        path:   path to the folder containing the Neurofinder data
        mode:   decide wether a single network is trained ('disco') or if 
                individual networks are trained for the five dataset series ('discos')
        gpu:    GPU used for training
        filename: name of the file in which the results are saved
        usedata: dataset series used for training, None = training on all dataset series
        
        '''
        self.path = path
        self.mode = mode
        self.usedata = usedata
        self.filename = filename
        self.dtype = torch.float
        if gpu is not None:
            self.device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.mode == 'disco':
            self.batch_size = 20
        else:
            if self.usedata == '00':
                self.batch_size = 6
            else:
                self.batch_size = 1
        self.learning_rate = 1e-4        
        self.epochs = 3000
        
        # get the dataset
        print('loading the datasets...')
        self.dataset = DISCoDataset(self.path,self.device,self.dtype, self.mode,
                                    self.usedata)
                    
        self.DISCoNet = DISCoNet(self.device).to(self.device)
        
    
    def pad_to_512(self,summary,correlations):
        if summary.size(3) < 512:
            diff = 512-summary.size(3)
            hdiff = int(diff/2)
            if diff % 2 == 0:
                pad = (hdiff,hdiff)
            else:
                pad = (hdiff,hdiff+1)
                            
            summary = F.pad(summary,pad,'constant',0)
            correlations = F.pad(correlations,pad,'constant',1)
            
        if summary.size(2) < 512:
            diff = 512-summary.size(2)
            hdiff = int(diff/2)
            if diff % 2 == 0:
                pad = (0,0,hdiff,hdiff)
            else:
                pad = (0,0,hdiff,hdiff+1)
            summary = F.pad(summary,pad,'constant',0)
            correlations = F.pad(correlations,pad,'constant',1)
            
        return(summary,correlations)
    
    def unpad(self,predicted,summary):
        if summary.size(3) < 512 and self.dataset.predict == True:
            diff = 512-summary.size(3)
            hdiff = int(diff/2)
            if diff % 2 == 0:
                predicted = predicted[:,:,:,hdiff:-hdiff]
            else:
                predicted = predicted[:,:,:,hdiff:-(hdiff+1)]
        if summary.size(2) < 512 and self.dataset.predict == True:
            diff = 512-summary.size(2)
            hdiff = int(diff/2)
            if diff % 2 == 0:
                predicted = predicted[:,:,hdiff:-hdiff]
            else:
                predicted = predicted[:,:,hdiff:-(hdiff+1)]
                    
        return(predicted)
        
    def train(self):
        try:
            # training loop:
            for epoch in range(self.epochs):
                    
                dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=0,drop_last=False)
                for i_batch, sample in enumerate(dataloader):
                    self.optimizer.zero_grad()   # zero the gradient buffers
                                    
                    # get ground truth
                    foreground = sample['label'].to(self.device)
                    affs = (1.-sample['affinities']).to(self.device)
                    target = torch.cat([affs,foreground],dim=1)
                    
                    # get input
                    corrs = sample['correlations'].to(self.device)
                    summ = sample['summary'].to(self.device)
                    
                    output = self.DISCoNet(corrs,summ)
                    
                    loss = self.criterion(output,target)
                    loss.backward()
                    self.optimizer.step()    # Does the update
                                
                    print(epoch, i_batch,"{:10.7f}".format(loss.item()))
                    
        except KeyboardInterrupt: 
            pass
        except: 
            raise 
        
        # save the really performed number of epochs, if training was terminated earlier
        self.epochs = epoch+1            
            
        return()
    
    def save_results(self,name,all_predictions):
        affs = 1.-all_predictions[0,:-1]
        fg = all_predictions[0,-1].view(1,affs.size(1),affs.size(2))
        bg = 1.-fg
                
        final = torch.cat([affs,bg,fg],dim=0)
        
        with h5.File(self.filename+'.h5','a') as o:
            if self.mode == 'disco':
                o.create_dataset(name,data=final.data.cpu().numpy(),compression='gzip')
            else:
                if self.usedata not in o:
                    grp = o.create_group(self.usedata)
                else:
                    grp = o[self.usedata]
                grp.create_dataset(name,data=final.data.cpu().numpy(),compression='gzip')
            
        return()
        
    def run(self):  
        
        # create the optimizer
        self.optimizer = optim.Adam(self.DISCoNet.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        
        # define loss function
        self.criterion = SorensenDiceLoss(channelwise=True)
        
        print('start training the DISCoNet...')
        self.train()    
        print('training terminated.\nsaving the results...')
        
        
        with torch.no_grad():
            # get ready for predicting...
            self.DISCoNet.eval()
            self.dataset.predict = True
            training_data_ = glob.glob(self.path+'neurofinder.*.h5')
            self.dataset.test_set = glob.glob(self.path+'neurofinder.*.test.h5')
            self.dataset.training_data = [item for item in training_data_ if item not in self.dataset.test_set]
            del self.dataset.data            
            
            # predict on train dataset
            print('predicting on the training data...')
            for v in range(len(self.dataset.training_data)):
                self.dataset.fetch_train_data(v)
                
                # predict over whole sequence
                sample = self.dataset.__getitem__(0)
                correlations = sample['correlations'][None,...].to(self.device) # add batch dimension
                summary = sample['summary'][None,...].to(self.device)
                name = sample['name']
                
                summary,correlations = self.pad_to_512(summary,correlations)
                predicted = self.DISCoNet(correlations,summary)
                predicted = self.unpad(predicted,summary)
                
                self.save_results(name,predicted)
                
                del self.dataset.data    
        
            # predict on test dataset
            print('predicting on the test data...')
            for v in range(len(self.dataset.test_set)):
                self.dataset.fetch_test_data(v)
                
                # predict over whole sequence
                sample = self.dataset.__getitem__(0)
                correlations = sample['correlations'][None,...].to(self.device) # add batch dimension
                summary = sample['summary'][None,...].to(self.device)
                name = sample['name']
                summary,correlations = self.pad_to_512(summary,correlations)
                predicted = self.DISCoNet(correlations,summary)
                predicted = self.unpad(predicted,summary)
                    
                self.save_results(name,predicted)
                    
                del self.dataset.data    
        
        print('predictions saved')
        return()

