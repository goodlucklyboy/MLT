import time
import matplotlib.pyplot as plt
import time
import os
import copy

import numpy as np



def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class loss_curve:
    def __init__(self, total_num):
        self.num = total_num
        self.train_loss = np.zeros(self.num)
        self.val_loss = np.zeros(self.num)
        self.train_acc = np.zeros(self.num)
        self.val_acc = np.zeros(self.num)

    def save_txt(self, name):
        result_txt = np.zeros([self.num, 4])
        result_txt[:, 0] = self.train_loss
        result_txt[:, 1] = self.val_loss
        result_txt[:, 2] = self.train_acc
        result_txt[:, 3] = self.val_acc
        np.savetxt(name, result_txt, fmt='%.4f')

    def display(self):
        plt.figure(1)
        plt.cla()
        plt.title('trainning curve')
        plt.xlabel('epoch num')
        plt.ylabel('loss value value')
        plt.plot(self.train_loss, 'r', label='train loss')
        plt.plot(self.val_loss, 'deeppink', label='val loss')
        plt.plot(self.train_acc, 'g', label='train accuracy')
        plt.plot(self.val_acc, 'greenyellow', label='val accuracy')
        plt.legend()
        # plt.show()
        plt.pause(0.01)

# def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
#     plt.figure(figsize=(10,10))
#     for i in range(n_classes):
#         inds = np.where(targets==i)[0]
#         plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     plt.legend(mnist_classes)
#     plt.pause(0.1)
#
# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if CUDA:
#                 images = images.cuda()
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labels

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_train_sample_siamese(data, label):
    images_so_far = 0
    img_0 = data[0]
    img_1 = data[1]
    labels = label

    sample_number = len(labels)
    for i in range(sample_number):

        images_so_far += 1
        ax = plt.subplot(1, sample_number, i+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        # ax.set_title('predicted: {} \n True_label: {}'.format(class_names[preds[j]],class_names[labels[j]]))
        plt.subplots_adjust(wspace =0.4,hspace=0.4)
        imshow(img_0.data[i])

        ax = plt.subplot(2, sample_number, i+sample_number+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        imshow(img_1.data[i])

def show_sample_classification(data, label):
    images_so_far = 0
    img = data
    labels = label

    sample_number = len(labels)
    for i in range(sample_number):

        images_so_far += 1
        ax = plt.subplot(2, sample_number//2, i+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        # ax.set_title('predicted: {} \n True_label: {}'.format(class_names[preds[j]],class_names[labels[j]]))
        plt.subplots_adjust(wspace =0.4,hspace=0.4)
        imshow(img.data[i])