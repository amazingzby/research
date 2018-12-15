import numpy as np

def split_npz():
    dataset = np.load('data/train.npz')
    total_num = dataset['imgs'].shape[0]
    slices = total_num//199
    for i in range(200):
        array ={}
        startId = i*slices
        endId   = (i+1)*slices
        if(endId>=total_num):
            endId = total_num
        array['imgs']        = dataset['imgs'][i*slices:(i+1)*slices]
        array['gtLandmarks'] = dataset['gtLandmarks'][i*slices:(i+1)*slices]
        array['filenames']   = dataset['filenames'][i*slices:(i+1)*slices]
        print('step '+str(i+1))
        print(startId)
        print(endId)
        np.savez('data/train_processed_'+str(i+1)+'.npz', **array)

def makeInitLandmark():
    initLandmark = np.load('data/train.npz')['initLandmarks'][0]
    initLandmark = np.reshape(initLandmark,(136))
    np.save('data/initlandmark.npy',initLandmark)
if __name__ == '__main__':
    split_npz()
    #makeInitLandmark()



