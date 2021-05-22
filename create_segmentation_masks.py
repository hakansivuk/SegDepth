import os
import cv2
import torchvision
import numpy as np
import torch
import time

def create_segmentation_masks(base_path):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
    model.eval()

    def get_gt(img_path):
        img = cv2.imread(img_path)
        im = torch.from_numpy(np.array([cv2.resize(img, (640,192), cv2.INTER_AREA).transpose(2,0,1)])/ 255.0)
        im = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im)
        predictions = model(im.float())
        return (np.argmax(predictions['out'].detach().numpy(), axis=1) > 0).astype(np.uint8)
        
    all_folders = ['2011_09_26']#os.listdir(base_path)
    elapsed_time = 0
    n_photos = 0
    for folder in all_folders:
        if folder != ".DS_Store":
            print(folder)
            all_sub_folders = os.listdir(base_path + '/' + folder)
            for sub_folder in all_sub_folders:
                if not sub_folder in ['calib_velo_to_cam.txt', '.DS_Store', 'calib_imu_to_velo.txt', 'calib_cam_to_cam.txt']:
                    print(sub_folder)
                    for im_folder in ['image_02','image_03']:
                        ims = os.listdir(base_path + '/' + folder + '/' + sub_folder + '/' + im_folder + '/data')
                        for im_path in ims:
                            if im_path.endswith('.png'):
                                s_time = time.time()
                                print(im_path)
                                im_name = im_path[:-4]
                                npy_path = base_path + '/' + folder + '/' + sub_folder + '/' + im_folder + '/data/' + im_name + '.npy'
                                if not os.path.isfile(npy_path):
                                    gt = get_gt(base_path + '/' + folder + '/' + sub_folder + '/' + im_folder + '/data/' + im_path)
                                    np.save(base_path + '/' + folder + '/' + sub_folder + '/' + im_folder + '/data/' + im_name, gt)
                                    #cv2.imwrite(base_path + '/' + folder + '/' + sub_folder + '/' + im_folder + '/data/' + im_name + '_resized1.png', gt.transpose(1,2,0) * 255)
                                    n_photos +=1 
                                    elapsed_time += time.time() - s_time
                                    print('Total elapsed time is {}'.format(elapsed_time))
                                    print('Total parsed photos is {}'.format(n_photos))
                                    elapsed_per_photo = elapsed_time / n_photos
                                    print('Elapsed time per photo is {}'.format(elapsed_per_photo))
                                else:
                                    elapsed_time += time.time() - s_time
                                

file_dir = os.path.dirname(__file__)

create_segmentation_masks(os.path.join(file_dir, "kitti_data"))

