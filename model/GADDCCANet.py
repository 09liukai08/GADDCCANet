import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy
import scipy.linalg
from sklearn import svm
from sklearn import neighbors, datasets

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.patch_sz = 7
        self.patch_sz2 = 7
        self.L1 = 8
        self.L2 = 8
        self.reg_term = 0.00001

        self.overlap_ratio = 0.5
        self.histblk_sz = 7
        self.num_class = np.power(2, self.L2)
        torch.backends.cudnn.enable = True

    def GetPatches(self, img, patch_sz, remove_mean):
        b, m, n = img.size()
        pad_sz =int((patch_sz - 1) / 2)
        dim = (pad_sz,pad_sz,pad_sz,pad_sz)
        pad_img = F.pad(img, dim, "constant", value=0) #matlab

        patches = torch.zeros([b, patch_sz, patch_sz, m * n]).cuda(img.get_device()) #[8,7,7,1024]
        for i in range(m):
            for j in range(n):
                patches[:,:,:, i * n + j] = pad_img[:, i:i+2*pad_sz+1, j:j+2*pad_sz+1]

        patches = patches.permute(2, 1, 0, 3).contiguous()
        patches = patches.view(patch_sz*patch_sz, b, -1)

        if remove_mean:
            patches = patches - patches.mean(0).view(1, b, -1) #matlab mean(patches,1)

        return patches

    def GetPatchesLayer2(self, img, patch_sz, remove_mean):
        b, l, m, n = img.size()  # [200, 4, 32, 32]
        pad_sz = int((patch_sz - 1) / 2)
        dim = (pad_sz, pad_sz, pad_sz, pad_sz)
        pad_img = F.pad(img, dim, "constant", value=0)  # matlab

        pad_img = pad_img.permute(3, 2, 1, 0).contiguous()  # [32, 32, 4, 8]

        # patches = torch.zeros([b, l, patch_sz, patch_sz, m * n]).cuda(img.get_device())  # [8,4,7,7,1024]
        patches = torch.zeros([patch_sz, patch_sz, l, b, m * n]).cuda(img.get_device())  # [8,4,7,7,1024]
        for i in range(m):
            for j in range(n):
                patches[:, :, :, :, i * n + j] = pad_img[i:i + 2 * pad_sz + 1, j:j + 2 * pad_sz + 1, :, :]

        # patches = patches.permute(3, 2, 1, 0, 4).contiguous() # [8,4,7,7,1024]-> [7,7,4,8,1024]
        # print(patches[:,:,0,0,:])
        patches = patches.view(patch_sz * patch_sz, l * b, -1)  # [49,4*8,1024]

        if remove_mean:
            patches = patches - patches.mean(0).view(1, l * b, -1)  # matlab mean(patches,1)

        return patches

    def GetPatchesForAllL1(self, img, patch_sz, remove_mean):
        b, l, m, n = img.size()  [200,4,32,32]
        pad_sz =int((patch_sz - 1) / 2)
        dim = (pad_sz,pad_sz,pad_sz,pad_sz)
        pad_img = F.pad(img, dim, "constant", value=0) #matlab

        patches = torch.zeros([b, l, patch_sz, patch_sz, m * n]).cuda(img.get_device()) #[8,7,7,1024]
        for i in range(m):
            for j in range(n):
                patches[:, :, :, :, i * n + j] = pad_img[:, :, i:i+2*pad_sz+1, j:j+2*pad_sz+1]


        patches = patches.permute(2, 1, 0, 3).contiguous()
        patches = patches.permute(3, 2, 1, 0, 4).contiguous()
        patches = patches.view(patch_sz*patch_sz, l*b, -1)

        if remove_mean:
            patches = patches - patches.mean(0).view(1, l*b, -1) #matlab mean(patches,1)

        return patches

    def DCCA_LowMemory(self, x, y, sample_labels, reg_term): #x: (b, self.patch_sz * self.patch_sz, -1)
        d1 = x.shape[0] # x[49, 1024*b]
        d2 = y.shape[0]

        num_samples = sample_labels.shape[1] #[204800]=b * m * n
        maxgrps = sample_labels.max()

        R12 = torch.zeros([d1, d2]).cuda(x.get_device())
        for i in range(maxgrps+1):
            iv = torch.full((1,num_samples), i).cuda(x.get_device()) #[1,204800]
            maskv= torch.eq(sample_labels, iv).squeeze() #[204800]
            masknon = maskv.nonzero().squeeze() #[5120]
            xi = x[:, masknon] #[49, 5120]
            xi1 = torch.sum(xi,dim=1).view(-1,1) #[49, 1]
            yi = y[:, masknon] #[49, 5120]
            yi1 = torch.sum(yi,dim=1).view(1,-1) #[1, 49]
            xys = torch.mm(xi1,yi1) #[49, 49]
            R12 = R12 + xys

        R11 = torch.mm(x, x.t())/num_samples #[49,49]
        R21 = R12.t()/num_samples #[49,49]
        #print(R21)
        R22 = torch.mm(y, y.t())/num_samples #[49,49]

        c1 = torch.cat((R11, R12), dim=1) #[49,98]
        c2 = torch.cat((R21, R22), dim=1) #[49,98]
        R0 = torch.cat((c1, c2), dim=0) #[98,98]

        D = torch.block_diag(R11, R22) #[98,98]

        while torch.matrix_rank(D)<D.shape[0]:
            D = D + reg_term*torch.mean(torch.abs(R0))*torch.eye(D.shape[0],D.shape[1]).cuda(x.get_device())

        D1, V1 = scipy.linalg.eig((R0-D).cpu(), D.cpu()) #D1[98]   V1[98,98]
        #print(D1)
        D1 = torch.from_numpy(D1).cuda(x.get_device())
        V1 = torch.from_numpy(V1).cuda(x.get_device())
        dvec = torch.real(D1) #[98]
        #dvec = torch.diag(D1)
        NV = torch.zeros([V1.shape[0],V1.shape[1]]).cuda(x.get_device()) #[98,98]
        dvec, index_dv = torch.sort(dvec)
        index_dv = torch.flipud(index_dv) #[98]
        ND = torch.zeros([D1.shape[0], D1.shape[0]]).cuda(x.get_device()) #[98,98]

        for i in range(D.shape[0]):
            ND[i, i] = D1[index_dv[i]]
            NV[:, i] = V1[:, index_dv[i]]

        eig1 = torch.div(NV[0:d1,:], torch.norm(NV[0:d1,:],dim=0)) #[49,98]
        eig2 = torch.div(NV[d1:d1+d2, :], torch.norm(NV[d1:d1+d2, :],dim=0)) #[49,98]
        return eig1, eig2

    def bin2dec(self, b, bits=7):
        #ask = 2 ** torch.arange(bits - 1, -1, -1)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device)
        return torch.sum(mask * b, -1)

    def entNoSum(self, x): #x: 7*7*b
        b, m, n = x.size()
        x = x.permute(2, 1, 0).long().contiguous()
        x = torch.reshape(x, (-1, b))
        grp = torch.zeros(x.size()).cuda(x.get_device()) #[49,b]

        for i in range(x.shape[0]):
            a0 = x[i,:] #(16) [0., 2., 2., 2., 2., 2., 0., 1., 0., 1., 2., 2., 2., 2., 2., 0.]])
            a1 = a0.repeat(x.shape[0], 1) #(16) ->(16,49)
            maskv = torch.eq(x, a1).long()
            grp[i,:] = torch.sum(maskv,dim=0)

        xhot = torch.nn.functional.one_hot(x, self.num_class).sum(dim=-1) #[49,b]
        total_sum = torch.sum(xhot, dim=0) #[b]

        # get entropy
        probs = grp / total_sum #[49,200]
        entropy = torch.mul(-probs, torch.log2(probs)) #[49,200]
        return entropy   #[7,7,200]


    def GetBlkEnt(self,img,blk_sz,num_bins,overlap_ratio):
        b, m, n = img.size()
        stride = torch.floor(torch.FloatTensor([self.histblk_sz - self.overlap_ratio * self.histblk_sz])).to(dtype=torch.int32)
        num_horz_blks = torch.floor(torch.tensor([(n - blk_sz) / stride + 1])).to(dtype=torch.int32)
        num_vert_blks = torch.floor(torch.tensor([(n - blk_sz) / stride + 1])).to(dtype=torch.int32)

        feature_vector = torch.zeros([num_vert_blks * num_horz_blks * num_bins,b]).cuda(img.get_device())
        count = 1

        for i in range(num_vert_blks):
            for j in range(num_horz_blks):
                blk = img[:, i*stride:i*stride+blk_sz, j*stride:j*stride+blk_sz] #blk [b,7,7]
                histvec = torch.zeros([num_bins,b]).cuda(img.get_device())
                eb = self.entNoSum(blk) #entblk: [7,7,b]
                blk1 = blk.permute(0, 2, 1).reshape(b, -1).t()
                histvec.scatter_(0, blk1.long(), eb) #why blk1 and eb on cpu now?
                feature_vector[(count - 1) * num_bins: count * num_bins, :] = histvec
                count = count + 1
        return feature_vector

    def GetBlkEntLayer2(self,img,blk_sz,num_bins,overlap_ratio):
        b, l, m, n = img.size()
        stride = torch.floor(torch.FloatTensor([self.histblk_sz - self.overlap_ratio * self.histblk_sz])).to(
            dtype=torch.int32)
        num_horz_blks = torch.floor(torch.tensor([(n - blk_sz) / stride + 1])).to(dtype=torch.int32)
        num_vert_blks = torch.floor(torch.tensor([(n - blk_sz) / stride + 1])).to(dtype=torch.int32)

        feature_vector = torch.zeros([num_vert_blks * num_horz_blks * num_bins, l, b]).cuda(img.get_device())

        img = img.permute(1, 0, 2, 3).contiguous().reshape(-1, m, n)
        # img = img.view(-1, m, n)
        count = 1
        for i in range(num_vert_blks):
            for j in range(num_horz_blks):
                blk = img[:, i * stride:i * stride + blk_sz, j * stride:j * stride + blk_sz]

                histvec = torch.zeros([num_bins, img.shape[0]]).cuda(img.get_device())
                eb = self.entNoSum(blk)  # entblk: [7,7,b]
                blk1 = blk.permute(0, 2, 1).reshape(blk.shape[0], -1).t()
                histvec.scatter_(0, blk1.long(), eb)  # why blk1 and eb on cpu now? [num_bins=16,b_batch]
                feature_vector[(count - 1) * num_bins: count * num_bins, :, :] = histvec.view(num_bins, l, b)
                count = count + 1

        return feature_vector

    def DDCAConv(self, train_view1, train_view2, patch_arr1_v1, patch_arr1_v2, train_labels, test_view1, test_view2, test_labels, b, m, n):
        b = train_view1.shape[0]  # b is  num_training_samples  # [8,1,32,32]
        #print(train_view1[0,0,:,:])
        bTest = test_view1.shape[0]  # b is  num_training_samples
        #bTest, mTest, nTest = test_view1.size()

        patch_arr1_labels = train_labels.view(b, 1).expand(b, m * n).contiguous()

        patch_arr1_v1 = patch_arr1_v1.view(self.patch_sz * self.patch_sz, -1)  # .contiguous()
        patch_arr1_v2 = patch_arr1_v2.view(self.patch_sz * self.patch_sz, -1)

        patch_arr1_labels = patch_arr1_labels.view(1, b * m * n)

        patch_arr1_v1 = patch_arr1_v1 - patch_arr1_v1.mean(1).view(-1,1) #[49, 1024*b]
        patch_arr1_v2 = patch_arr1_v2 - patch_arr1_v2.mean(1).view(-1,1)

        eigvec1_v1, eigvec1_v2 = self.DCCA_LowMemory(patch_arr1_v1, patch_arr1_v2, patch_arr1_labels, self.reg_term)
        #eigvec1_v1 = ReadMat('eigvec1_v1.mat').cuda(train_view1.get_device())
        #eigvec1_v2 = ReadMat('eigvec1_v2.mat').cuda(train_view1.get_device())

        filter_maps1_v1 = eigvec1_v1.view(1, self.patch_sz, self.patch_sz, -1)[:,:,:,0:self.L1] #[1,7,7,98]
        filter_maps1_v2 = eigvec1_v2.view(1, self.patch_sz, self.patch_sz, -1)[:,:,:,0:self.L1]

        #print(filter_maps1_v1[0,:,:,0])
        filter_maps1_v1 = filter_maps1_v1.permute(3, 0, 2, 1).contiguous().to(dtype=torch.float32)  #[98,1,7,7]   self.L1=4
        filter_maps1_v2 = filter_maps1_v2.permute(3, 0, 2, 1).contiguous().to(dtype=torch.float32)
        filter_maps1_v1 = torch.rot90(filter_maps1_v1, -2, [2, 3])
        filter_maps1_v2 = torch.rot90(filter_maps1_v2, -2, [2, 3])
        #print(filter_maps1_v1)
        # train_view1 = train_view1.view(b,1,m,n) #[8,1,32,32]
        # train_view2 = train_view2.view(b,1,m,n)
        # test_view1 = test_view1.view(bTest, 1, mTest, nTest)  # [8,1,32,32]
        # test_view2 = test_view2.view(bTest, 1, mTest, nTest)

        real_batch_size = 200
        splits = math.ceil(b / real_batch_size)
        real_batch_size_forTest = bTest // splits

        for i in range(splits):
            #print(str(i))
            left = i * real_batch_size
            leftTest = i * real_batch_size_forTest
            if i == splits-1:
                right = b
                rightTest = bTest
            else:
                right = left + real_batch_size
                rightTest = leftTest + real_batch_size_forTest
            #print(str(left)+'-'+str(right))
            #print(str(leftTest)+'-'+str(rightTest))
            batch_train_act_arr1_v1 = F.conv2d(train_view1[left:right], filter_maps1_v1,padding='same')
            batch_train_act_arr1_v2 = F.conv2d(train_view2[left:right], filter_maps1_v2, padding='same')
            batch_test_act_arr1_v1 = F.conv2d(test_view1[leftTest:rightTest], filter_maps1_v1, padding='same')
            batch_test_act_arr1_v2 = F.conv2d(test_view2[leftTest:rightTest], filter_maps1_v2, padding='same')
            if i == 0:
                train_act_arr1_v1 = batch_train_act_arr1_v1
                train_act_arr1_v2 = batch_train_act_arr1_v2
                test_act_arr1_v1 = batch_test_act_arr1_v1
                test_act_arr1_v2 = batch_test_act_arr1_v2
            else:
                train_act_arr1_v1 = torch.cat((train_act_arr1_v1, batch_train_act_arr1_v1), 0)
                train_act_arr1_v2 = torch.cat((train_act_arr1_v2, batch_train_act_arr1_v2), 0)
                test_act_arr1_v1 = torch.cat((test_act_arr1_v1, batch_test_act_arr1_v1), 0)
                test_act_arr1_v2 = torch.cat((test_act_arr1_v2, batch_test_act_arr1_v2), 0)

        #for i in range(self.L1):
        return train_act_arr1_v1, train_act_arr1_v2, test_act_arr1_v1, test_act_arr1_v2

    def PCA(self,feats,num_pca_terms):
        feats_mean= torch.mean(feats,0)
        feats = feats - feats_mean.expand_as(feats)
        U,S,V = torch.svd(torch.t(feats))


    def GetFeatureForClassification(self,view1, view2, labels, testview1, testview2, testlabels):
        b, m, n = view1.size()  # b is  num_training_samples

        bTest, mTest, nTest = testview1.size()

        real_batch_size = 2048
        splits = math.ceil(b / real_batch_size)
        real_batch_size_forTest = bTest // splits

        maxgrps = labels.max()
        # print(maxgrps)
        R12 = torch.zeros([self.patch_sz * self.patch_sz, self.patch_sz * self.patch_sz]).cuda(view1.get_device())
        for j in range(maxgrps + 1):
            #print('maxgrps'+str(j))
            for i in range(splits):
                #print(str(i))
                left = i * real_batch_size
                if i == splits - 1:
                    right = b
                else:
                    right = left + real_batch_size
                #print(str(left) + '-' + str(right))
                batch_train_act_arr1_v1, batch_train_act_arr1_v2, batch_train_labels = view1[left:right], view2[
                                                                                                          left:right], labels[
                                                                                                                       left:right]

                batch_patch_arr1_v1 = self.GetPatches(batch_train_act_arr1_v1, self.patch_sz, 1)  # [49,4*8,1024]
                batch_patch_arr1_v2 = self.GetPatches(batch_train_act_arr1_v2, self.patch_sz, 1)


                # print(batch_patch_arr1_v1[:, 0, 0])

                patch_arr1_labels = batch_train_labels.view(-1, 1).expand(-1, m * n).contiguous()

                batch_patch_arr1_v1 = batch_patch_arr1_v1.view(self.patch_sz * self.patch_sz, -1)  # .contiguous()
                batch_patch_arr1_v2 = batch_patch_arr1_v2.view(self.patch_sz * self.patch_sz, -1)

                patch_arr1_labels = patch_arr1_labels.view(1, -1)
                # print(patch_arr1_labels)

                batch_patch_arr1_v1 = batch_patch_arr1_v1 - batch_patch_arr1_v1.mean(1).view(-1, 1)  # [49, 1024*b]
                batch_patch_arr1_v2 = batch_patch_arr1_v2 - batch_patch_arr1_v2.mean(1).view(-1, 1)

                num_samples = patch_arr1_labels.shape[1]  # [204800]=b * m * n
                #print("num_samples: " + str(num_samples))

                # print(batch_patch_arr1_v1.shape)
                if j == 0:
                    # print(num_samples)
                    if i == 0:
                        R11 = torch.mm(batch_patch_arr1_v1, batch_patch_arr1_v1.t()) / num_samples
                        R22 = torch.mm(batch_patch_arr1_v2, batch_patch_arr1_v2.t()) / num_samples
                        # print(R11)
                    else:
                        batch_R11 = torch.mm(batch_patch_arr1_v1, batch_patch_arr1_v1.t()) / num_samples
                        R11 = R11 + batch_R11
                        batch_R22 = torch.mm(batch_patch_arr1_v2, batch_patch_arr1_v2.t()) / num_samples
                        R22 = R22 + batch_R22

                iv = torch.full((1, num_samples), j).cuda(view1.get_device())

                maskv = torch.eq(patch_arr1_labels, iv).squeeze()  # [204800]

                masknon = maskv.nonzero().squeeze()  # [5120]

                xi = batch_patch_arr1_v1[:, masknon]  # [49, 5120]
                #print("xi: " + str(xi.size()))
                batch_xi1 = torch.sum(xi, dim=1).view(-1, 1)  # [49, 1]
                yi = batch_patch_arr1_v2[:, masknon]  # [49, 5120]
                batch_yi1 = torch.sum(yi, dim=1).view(1, -1)  # [1, 49]

                if i == 0:
                    xi1 = batch_xi1
                    yi1 = batch_yi1
                else:
                    xi1 = torch.cat((xi1, batch_xi1), 1)
                    yi1 = torch.cat((yi1, batch_yi1), 0)
            xys = torch.mm(xi1, yi1)
            R12 = R12 + xys

        R21 = R12.t() / num_samples  # [49,49]

        c1 = torch.cat((R11, R12), dim=1)  # [49,98]
        c2 = torch.cat((R21, R22), dim=1)  # [49,98]
        R0 = torch.cat((c1, c2), dim=0)  # [98,98]

        D = torch.block_diag(R11, R22)  # [98,98]

        while torch.matrix_rank(D) < D.shape[0]:
            D = D + self.reg_term * torch.mean(torch.abs(R0)) * torch.eye(D.shape[0], D.shape[1]).cuda(
                view1.get_device())

        D1, V1 = scipy.linalg.eig((R0 - D).cpu(), D.cpu())  # D1[98]   V1[98,98]
        # print(D1)
        D1 = torch.from_numpy(D1).cuda(labels.get_device())
        V1 = torch.from_numpy(V1).cuda(labels.get_device())
        dvec = torch.real(D1)  # [98]
        # dvec = torch.diag(D1)
        NV = torch.zeros([V1.shape[0], V1.shape[1]]).cuda(labels.get_device())  # [98,98]
        dvec, index_dv = torch.sort(dvec)
        index_dv = torch.flipud(index_dv)  # [98]
        ND = torch.zeros([D1.shape[0], D1.shape[0]]).cuda(labels.get_device())  # [98,98]

        for i in range(D.shape[0]):
            ND[i, i] = D1[index_dv[i]]
            NV[:, i] = V1[:, index_dv[i]]

        d1 = self.patch_sz * self.patch_sz
        d2 = self.patch_sz * self.patch_sz
        eigvec1_v1 = torch.div(NV[0:d1, :], torch.norm(NV[0:d1, :], dim=0))  # [49,98]
        eigvec1_v2 = torch.div(NV[d1:d1 + d2, :], torch.norm(NV[d1:d1 + d2, :], dim=0))  # [49,98]

        filter_maps1_v1 = eigvec1_v1.view(1, self.patch_sz, self.patch_sz, -1)[:, :, :, 0:self.L1]  # [1,7,7,98]
        filter_maps1_v2 = eigvec1_v2.view(1, self.patch_sz, self.patch_sz, -1)[:, :, :, 0:self.L1]

        filter_maps1_v1 = filter_maps1_v1.permute(3, 0, 2, 1).contiguous().to(dtype=torch.float32)  # [98,1,7,7]   self.L1=4
        filter_maps1_v2 = filter_maps1_v2.permute(3, 0, 2, 1).contiguous().to(dtype=torch.float32)

        for i in range(splits):
            # print(str(i))
            left = i * real_batch_size
            if i == splits - 1:
                right = b
            else:
                right = left + real_batch_size
            #print(str(left) + '-' + str(right))

            # print("view1: " + str(view1.size())) #[b,m,n]
            batch_train_act_arr1_v1 = F.conv2d(view1[left:right].view(-1, 1, m, n),
                                               filter_maps1_v1.to(dtype=torch.float32),
                                               padding='same')
            batch_train_act_arr1_v2 = F.conv2d(view2[left:right].view(-1, 1, m, n),
                                               filter_maps1_v2.to(dtype=torch.float32),
                                               padding='same')

            if i == 0:
                view1Res = batch_train_act_arr1_v1
                view2Res = batch_train_act_arr1_v2
            else:

                view1Res = torch.cat((view1Res, batch_train_act_arr1_v1), 0)
                view2Res = torch.cat((view2Res, batch_train_act_arr1_v2), 0)

        # loop for testing data
        splitsForTestData = math.ceil(bTest / real_batch_size)
        for i in range(splitsForTestData):
            # print(str(i))
            leftTest = i * real_batch_size
            if i == splitsForTestData - 1:
                rightTest = bTest
            else:
                rightTest = leftTest + real_batch_size
            #print(str(leftTest) + '-' + str(rightTest))

            batch_test_act_arr1_v1 = F.conv2d(testview1[leftTest:rightTest].view(-1, 1, mTest, nTest),
                                              filter_maps1_v1.to(dtype=torch.float32),
                                              padding='same')
            batch_test_act_arr1_v2 = F.conv2d(testview2[leftTest:rightTest].view(-1, 1, mTest, nTest),
                                              filter_maps1_v2.to(dtype=torch.float32),
                                              padding='same')

            if i == 0:
                testview1Res = batch_test_act_arr1_v1
                testview2Res = batch_test_act_arr1_v2
            else:
                testview1Res = torch.cat((testview1Res, batch_test_act_arr1_v1), 0)
                testview2Res = torch.cat((testview2Res, batch_test_act_arr1_v2), 0)

        view1 = view1Res
        view2 = view2Res
        testview1 = testview1Res
        testview2 = testview2Res

        #print('layer2')

        #layer2
        self.patch_sz =7
        torch.cuda.empty_cache()
        real_batch_size = 128
        splits = math.ceil(b / real_batch_size)
        real_batch_size_forTest = bTest // splits
        R12 = torch.zeros([self.patch_sz * self.patch_sz, self.patch_sz * self.patch_sz]).cuda(batch_patch_arr1_v1.get_device())
        for j in range(maxgrps + 1):
            #print('layer2 maxgrps' + str(j))
            for i in range(splits):
                # print(str(i))
                left = i * real_batch_size
                if i == splits - 1:
                    right = b
                else:
                    right = left + real_batch_size
                #print(str(left) + '-' + str(right))
                batch_train_act_arr1_v1, batch_train_act_arr1_v2, batch_train_labels = view1[left:right], view2[left:right], labels[left:right]

                batch_patch_arr1_v1 = self.GetPatchesLayer2(batch_train_act_arr1_v1, self.patch_sz, 1)  #[49,4*8,1024]
                batch_patch_arr1_v2 = self.GetPatchesLayer2(batch_train_act_arr1_v2, self.patch_sz, 1)

                patch_arr1_labels = batch_train_labels.view(-1, 1).expand(-1, self.L1*m * n).contiguous()

                batch_patch_arr1_v1 = batch_patch_arr1_v1.view(self.patch_sz * self.patch_sz, -1)  # .contiguous()
                batch_patch_arr1_v2 = batch_patch_arr1_v2.view(self.patch_sz * self.patch_sz, -1)

                patch_arr1_labels = patch_arr1_labels.view(1, -1)

                batch_patch_arr1_v1 = batch_patch_arr1_v1 - batch_patch_arr1_v1.mean(1).view(-1, 1)  # [49, 1024*b]
                batch_patch_arr1_v2 = batch_patch_arr1_v2 - batch_patch_arr1_v2.mean(1).view(-1, 1)

                num_samples = patch_arr1_labels.shape[1]  # [204800]=b * m * n

                if j == 0:
                    if i == 0:
                        R11 = torch.mm(batch_patch_arr1_v1, batch_patch_arr1_v1.t())/ num_samples
                        R22 = torch.mm(batch_patch_arr1_v2, batch_patch_arr1_v2.t())/ num_samples
                    else:
                        batch_R11 = torch.mm(batch_patch_arr1_v1, batch_patch_arr1_v1.t())/ num_samples
                        R11 = R11 + batch_R11
                        batch_R22 = torch.mm(batch_patch_arr1_v2, batch_patch_arr1_v2.t())/ num_samples
                        R22 = R22 + batch_R22

                iv = torch.full((1, num_samples), j).cuda(batch_patch_arr1_v1.get_device())  # [1,204800]
                maskv = torch.eq(patch_arr1_labels, iv).squeeze()  # [204800]
                masknon = maskv.nonzero().squeeze()  # [5120]
                xi = batch_patch_arr1_v1[:, masknon]  # [49, 5120]
                batch_xi1 = torch.sum(xi, dim=1).view(-1, 1)  # [49, 1]
                yi = batch_patch_arr1_v2[:, masknon]  # [49, 5120]
                batch_yi1 = torch.sum(yi, dim=1).view(1, -1)  # [1, 49]

                if i == 0:
                    xi1 = batch_xi1
                    yi1 = batch_yi1
                else:
                    xi1 = torch.cat((xi1, batch_xi1), 1)
                    yi1 = torch.cat((yi1, batch_yi1), 0)
            xys = torch.mm(xi1, yi1)
            R12 = R12 + xys

        R21 = R12.t()/num_samples # [49,49]

        c1 = torch.cat((R11, R12), dim=1)  # [49,98]
        c2 = torch.cat((R21, R22), dim=1)  # [49,98]
        R0 = torch.cat((c1, c2), dim=0)  # [98,98]

        D = torch.block_diag(R11, R22)  # [98,98]

        while torch.matrix_rank(D) < D.shape[0]:
            D = D + self.reg_term * torch.mean(torch.abs(R0)) * torch.eye(D.shape[0], D.shape[1]).cuda(view1.get_device())

        D1, V1 = scipy.linalg.eig((R0 - D).cpu(), D.cpu())  # D1[98]   V1[98,98]
        # print(D1)
        D1 = torch.from_numpy(D1).cuda(labels.get_device())
        V1 = torch.from_numpy(V1).cuda(labels.get_device())
        dvec = torch.real(D1)  # [98]
        # dvec = torch.diag(D1)
        NV = torch.zeros([V1.shape[0], V1.shape[1]]).cuda(labels.get_device())  # [98,98]
        dvec, index_dv = torch.sort(dvec)
        index_dv = torch.flipud(index_dv)  # [98]
        ND = torch.zeros([D1.shape[0], D1.shape[0]]).cuda(labels.get_device())  # [98,98]

        for i in range(D.shape[0]):
            ND[i, i] = D1[index_dv[i]]
            NV[:, i] = V1[:, index_dv[i]]

        d1 = self.patch_sz * self.patch_sz
        d2 = self.patch_sz * self.patch_sz
        eigvec1_v1 = torch.div(NV[0:d1, :], torch.norm(NV[0:d1, :], dim=0))  # [49,98]
        eigvec1_v2 = torch.div(NV[d1:d1 + d2, :], torch.norm(NV[d1:d1 + d2, :], dim=0))  # [49,98]

        filter_maps1_v1 = eigvec1_v1.view(1, self.patch_sz, self.patch_sz, -1)[:, :, :, 0:self.L2]  # [1,7,7,98]
        filter_maps1_v2 = eigvec1_v2.view(1, self.patch_sz, self.patch_sz, -1)[:, :, :, 0:self.L2]

        filter_maps1_v1 = filter_maps1_v1.permute(3, 0, 2, 1).contiguous()  # [98,1,7,7]   self.L1=4
        filter_maps1_v2 = filter_maps1_v2.permute(3, 0, 2, 1).contiguous()

        torch.cuda.empty_cache()
        stride = torch.floor(torch.FloatTensor([self.histblk_sz - self.overlap_ratio * self.histblk_sz])).to(dtype=torch.int32)
        num_horz_blks = torch.floor(torch.tensor([(n - self.histblk_sz) / stride + 1])).to(dtype=torch.int32)
        num_vert_blks = torch.floor(torch.tensor([(n - self.histblk_sz) / stride + 1])).to(dtype=torch.int32)
        num_bins = torch.pow(torch.tensor([2]), self.L2)  # num_bins: 16

        real_batch_size = 328
        splits = math.ceil(b / real_batch_size)

        feature_trainDate = torch.zeros((num_vert_blks * num_horz_blks * num_bins * self.L1 *2, view1.shape[0]))
        for i in range(splits):
            # print(str(i))
            left = i * real_batch_size
            if i == splits - 1:
                right = b
            else:
                right = left + real_batch_size
            #print(str(left) + '-' + str(right))

            train_act_arr1_v1_res = F.conv2d(view1[left:right].view(-1, 1, m, n), filter_maps1_v1.to(dtype=torch.float32),
                                               padding='same')
            train_act_arr1_v2_res = F.conv2d(view2[left:right].view(-1, 1, m, n), filter_maps1_v2.to(dtype=torch.float32),
                                               padding='same')

            train_act_arr1_v1_res = train_act_arr1_v1_res.view(-1, self.L1, self.L2, m, n) #[batch, 4, 4, 32, 32]
            train_act_arr1_v2_res = train_act_arr1_v2_res.view(-1, self.L1, self.L2, m, n)

            train_act_arr1_v1_res = train_act_arr1_v1_res - train_act_arr1_v1_res.mean(4).mean(3).view(train_act_arr1_v1_res.shape[0], train_act_arr1_v1_res.shape[1], train_act_arr1_v1_res.shape[2],
                                                       1,
                                                       1)  # train_act_arr1_v1[8,4,32,32]
            train_act_arr1_v2_res = train_act_arr1_v2_res - train_act_arr1_v2_res.mean(4).mean(3).view(train_act_arr1_v2_res.shape[0], train_act_arr1_v2_res.shape[1], train_act_arr1_v2_res.shape[2],
                                                       1, 1)

            train_act_arr1_v1_res = torch.gt(train_act_arr1_v1_res, 0)  # (batch,L1, L2, 32,32)
            train_act_arr1_v2_res = torch.gt(train_act_arr1_v2_res, 0)

            train_act_arr1_v1_res = train_act_arr1_v1_res.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.L2)
            train_act_arr1_v2_res = train_act_arr1_v2_res.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.L2)
            view1Res = self.bin2dec(train_act_arr1_v1_res, self.L2).view(-1, self.L1, m, n)
            view2Res = self.bin2dec(train_act_arr1_v2_res, self.L2).view(-1, self.L1, m, n)
            train_act_arr1_v1_res = self.GetBlkEntLayer2(view1Res, self.histblk_sz, num_bins, self.overlap_ratio)  # (1296,4,8)
            train_act_arr1_v2_res = self.GetBlkEntLayer2(view2Res, self.histblk_sz, num_bins, self.overlap_ratio)
            train_act_arr1_v1_res = train_act_arr1_v1_res.permute(1, 0, 2).contiguous().reshape(-1, right-left)  # train_feature 应该是正小数，why我算的是负整数
            train_act_arr1_v2_res = train_act_arr1_v2_res.permute(1, 0, 2).contiguous().reshape(-1, right-left)
            feature_trainDate[:, left:right] = torch.cat((train_act_arr1_v1_res, train_act_arr1_v2_res), 0).cpu()  # [2592,200]


        dim_keep = 199
        u, s, prin_comp = torch.pca_lowrank(feature_trainDate.t(), q=dim_keep)
        feature_trainDate = torch.mm(prin_comp.t(), feature_trainDate)  # (199,b)
        train_feature = feature_trainDate.t()  # (b,199)

        test_feature = torch.zeros((dim_keep, testview1.shape[0]))
        splitsForTestData = math.ceil(bTest / real_batch_size)
        for i in range(splitsForTestData):
            # print(str(i))
            leftTest = i * real_batch_size
            if i == splitsForTestData - 1:
                rightTest = bTest
            else:
                rightTest = leftTest + real_batch_size
            #print(str(leftTest) + '-' + str(rightTest))

            train_act_arr1_v1_res = F.conv2d(testview1[leftTest:rightTest].view(-1, 1, mTest, nTest), filter_maps1_v1.to(dtype=torch.float32),
                                              padding='same')
            train_act_arr1_v2_res = F.conv2d(testview2[leftTest:rightTest].view(-1, 1, mTest, nTest), filter_maps1_v2.to(dtype=torch.float32),
                                              padding='same')

            train_act_arr1_v1_res = train_act_arr1_v1_res.view(-1, self.L1, self.L2, m, n)
            train_act_arr1_v2_res = train_act_arr1_v2_res.view(-1, self.L1, self.L2, m, n)

            train_act_arr1_v1_res = train_act_arr1_v1_res - train_act_arr1_v1_res.mean(4).mean(3).view(train_act_arr1_v1_res.shape[0], train_act_arr1_v1_res.shape[1], train_act_arr1_v1_res.shape[2],
                                                                   1,
                                                                   1)
            train_act_arr1_v2_res = train_act_arr1_v2_res - train_act_arr1_v2_res.mean(4).mean(3).view(train_act_arr1_v2_res.shape[0], train_act_arr1_v2_res.shape[1], train_act_arr1_v2_res.shape[2],
                                                                   1, 1)

            train_act_arr1_v1_res = torch.gt(train_act_arr1_v1_res, 0)
            train_act_arr1_v2_res = torch.gt(train_act_arr1_v2_res, 0)

            train_act_arr1_v1_res = train_act_arr1_v1_res.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.L2)
            train_act_arr1_v2_res = train_act_arr1_v2_res.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.L2)
            view1Res = self.bin2dec(train_act_arr1_v1_res, self.L2).view(-1, self.L1, m, n)
            view2Res = self.bin2dec(train_act_arr1_v2_res, self.L2).view(-1, self.L1, m, n)
            train_act_arr1_v1_res = self.GetBlkEntLayer2(view1Res, self.histblk_sz, num_bins, self.overlap_ratio)
            train_act_arr1_v2_res = self.GetBlkEntLayer2(view2Res, self.histblk_sz, num_bins, self.overlap_ratio)
            train_act_arr1_v1_res = train_act_arr1_v1_res.permute(1, 0, 2).contiguous().reshape(-1, rightTest-leftTest)  # test_feature 应该是正小数，why我算的是负整数
            train_act_arr1_v2_res = train_act_arr1_v2_res.permute(1, 0, 2).contiguous().reshape(-1, rightTest-leftTest)
            train_act_arr1_v1_res = torch.cat((train_act_arr1_v1_res, train_act_arr1_v2_res), 0).cpu()
            test_feature[:, leftTest:rightTest] = torch.mm(prin_comp.t(), train_act_arr1_v1_res)

        test_feature = test_feature.t()  # (b,199)

        return train_feature, test_feature

    def forward(self, train_view1, train_view2, train_labels, test_view1, test_view2, test_labels):

        bTest, mTest, nTest = test_view1.size()
        #print(test_labels)

        train_view1, test_view1 = self.GetFeatureForClassification(train_view1, train_view2, train_labels, test_view1, test_view2, test_labels)

        clf = svm.SVC(decision_function_shape='ovr', kernel='poly', gamma="auto") 

        clf.fit(train_view1, train_labels.cpu())
        res1 = clf.predict(test_view1)

        count = 0
        for i in range(bTest):
            if res1[i] == test_labels[i]:
                count += 1
        print('accSVM:' + str(count/bTest)) #10605

        return 0

