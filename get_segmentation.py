import numpy as np
import h5py as h5

from GASP.segmentation import GaspFromAffinities

START_AGGLO_FROM_WSDT_SUPERPIXELS = False

# affinity offsets
# add artificial 3rd dimension
OFFSETS = [[0,-1, 0], [0,0, -1],
           [0,-5, 0], [0,0, -5], [0,-5, -5]]


def get_mask(affs,OFFSETS,bg,maskthreshold):
    mask = np.ones(affs.shape)
    bg_mask = np.ones((1,affs.shape[2],affs.shape[3]))
    bg_mask[np.where(bg > maskthreshold)] = 0.
    for o,offs in enumerate(OFFSETS):
        mask[o] = bg_mask
        off_d, off_x, off_y = offs
        if off_x < 0 and off_y == 0:
            mask[o,:,-off_x:] = bg_mask[:,:off_x]
        elif off_x == 0 and off_y < 0:
            mask[o,:,:,-off_y:] = bg_mask[:,:,:off_y]
        elif off_x < 0 and off_y < 0:
            mask[o,:,-off_x:,-off_y:] = bg_mask[:,:off_x,:off_y]
        
    return(mask)
 
class BackgroundLabelSuperpixelGenerator(object):
    def __call__(self, affinities, foreground_mask):
        pixel_segm = np.arange(np.prod(foreground_mask.shape), dtype='uint64').reshape(foreground_mask.shape) + 1
        return (pixel_segm * foreground_mask).astype('int64') - 1

        
def get_segmentation(prediction_file,mode,usedata):    
    out_file = prediction_file + '_gasp'            
    if mode == 'disco':        
        # load affinities and invert the repulsive channels
        with h5.File(prediction_file+'.h5', "r") as f:
            for name in f.keys():
                        
                affs = f[name][...]
                fg = affs[-2].reshape(1,affs.shape[1],affs.shape[2])
                bg = affs[-1].reshape(1,affs.shape[1],affs.shape[2])
                affs = affs[:-2]    
                affs = affs.reshape(affs.shape[0],1,affs.shape[1],affs.shape[2])
                    
                # Run GASP:
                run_GASP_kwargs = {'linkage_criteria': 'average',
                                   'add_cannot_link_constraints': False}
                superpixel_gen = BackgroundLabelSuperpixelGenerator()
                gasp_instance = GaspFromAffinities(OFFSETS,
                                                   superpixel_generator=superpixel_gen,
                                                   run_GASP_kwargs=run_GASP_kwargs)
                        
                fg_mask = np.ones(fg.shape)    # construct foreground mask
                fg_mask[np.where(bg>fg)] = 0.  # all pixels which are for sure background are excluded from clustering
                final_segmentation, runtime = gasp_instance(affs,fg_mask)
                        
                final = np.concatenate((final_segmentation,fg),axis=0)
                with h5.File(out_file+".h5", "a") as g:
                    g.create_dataset(name, data=final, compression="gzip")
                        
    elif mode == 'discos':        
        # load affinities and invert the repulsive channels
        with h5.File(prediction_file+'.h5', "r") as f:
            f_u = f[usedata]
            for name in f_u.keys():
                        
                affs = f_u[name][...]
                fg = affs[-2].reshape(1,affs.shape[1],affs.shape[2])
                bg = affs[-1].reshape(1,affs.shape[1],affs.shape[2])
                affs = affs[:-2]    
                affs = affs.reshape(affs.shape[0],1,affs.shape[1],affs.shape[2])
                    
                # Run GASP:
                run_GASP_kwargs = {'linkage_criteria': 'average',
                                   'add_cannot_link_constraints': False}
                superpixel_gen = BackgroundLabelSuperpixelGenerator()
                gasp_instance = GaspFromAffinities(OFFSETS,
                                                   superpixel_generator=superpixel_gen,
                                                   run_GASP_kwargs=run_GASP_kwargs)
                        
                fg_mask = np.ones(fg.shape)    # construct foreground mask
                fg_mask[np.where(bg>fg)] = 0.  # all pixels which are for sure background are excluded from clustering
                final_segmentation, runtime = gasp_instance(affs,fg_mask)
                        
                final = np.concatenate((final_segmentation,fg),axis=0)
                with h5.File(out_file+".h5", "a") as g:
                    if usedata in g:
                        g[usedata].create_dataset(name, data=final, compression="gzip")
                    else:
                        grp = g.create_group(usedata)
                        grp.create_dataset(name, data=final, compression="gzip")
                        
    return(out_file)
                        
                        
