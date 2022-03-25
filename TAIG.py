import os
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import torchvision
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
from torch.backends import cudnn
import numpy as np
import pretrainedmodels
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_dir', type=str, default = 'test/')
parser.add_argument('--target_attack', default=False, action='store_true')
parser.add_argument('--s_num', type=str, default='20')
parser.add_argument('--r_flag', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='senet')
args = parser.parse_args()

# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)



def compute_ig(inputs,label_inputs,model):
    baseline = np.zeros(inputs.shape)
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in
                     range(0, steps + 1)]
    scaled_inputs = np.asarray(scaled_inputs)
    if r_flag==True:
        scaled_inputs = scaled_inputs + np.random.uniform(-epsilon,epsilon,scaled_inputs.shape)
    scaled_inputs = torch.from_numpy(scaled_inputs)
    scaled_inputs = scaled_inputs.to(device, dtype=torch.float)
    scaled_inputs.requires_grad_(True)
    att_out = model(scaled_inputs)
    score = att_out[:, label_inputs]
    loss = -torch.mean(score)
    model.zero_grad()
    loss.backward()
    grads = scaled_inputs.grad.data
    avg_grads = torch.mean(grads, dim=0)
    delta_X = scaled_inputs[-1] - scaled_inputs[0]
    integrated_grad = delta_X * avg_grads
    IG = integrated_grad.cpu().detach().numpy()
    del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
    return IG


if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.makedirs(args.save_dir, exist_ok=True)
    epsilon = args.epsilon/255
    batch_size = args.batch_size
    save_dir = args.save_dir
    niters = args.niters
    target_attack = args.target_attack
    r_flag = args.r_flag
    s_num = int(args.s_num)
    model_name = args.model_name

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    trans = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor()
    ])
    dataset = SelectedImagenet(imagenet_val_dir='data/imagenet/ILSVRC2012_img_val',
                               selected_images_csv='data/imagenet/selected_imagenet.csv',
                               transform=trans
                               )
    ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)


    if model_name == 'inception':
        model = torchvision.models.inception_v3(pretrained=True)
    elif model_name == 'senet':
        model =  pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet':
        model = torchvision.models.densenet121(pretrained=True)
    else:
        print('No implemation')
    if model_name in ['resnet', 'densenet', 'senet']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        input_size = [3, 224, 224]
    else:
        input_size = [3, 299, 299]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    norm = T.Normalize(tuple(mean), tuple(std))
    resize = T.Resize(tuple((input_size[1:])))
    model = nn.Sequential(
        resize,
        norm,
        model
    )
    model.eval()
    model.to(device)
    if target_attack:
        label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
    label_ls = []
    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]
        ori_img = ori_img.to(device)
        img = ori_img.clone()
        m = 0
        for i in range(niters):
            img_x = img
            img_x.requires_grad_(True)
            steps = s_num
            igs = []
            for im_i in range(list(img_x.shape)[0]):
                inputs = img_x[im_i].cpu().detach().numpy()
                label_inputs = label[im_i]
                integrated_grad = compute_ig(inputs, label_inputs, model)
                igs.append(integrated_grad)
            igs = np.array(igs)

            model.zero_grad()
            input_grad=torch.from_numpy(igs)
            input_grad=input_grad.cuda()

            if target_attack:
                input_grad = - input_grad

            img = img.data + 1./255 * torch.sign(input_grad)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data*255).cpu().numpy().astype(np.uint8()))
        del img, ori_img, input_grad
        print('batch_{}.npy saved'.format(ind))

    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('images saved')
