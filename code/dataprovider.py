from anytree import NodeMixin, iterators, RenderTree
import math
#from pyzem.swc import swc
import swc
import numpy as np
import os
import os.path

import scipy.misc

level_limit = 112

class dataprovider(object):

    def __init__(self):
        self.input = None

    def load(self,tree):
        self.rawdata = tree

    def maxbranchorder(self):
        niter = iterators.LevelOrderIter(self.rawdata._root)
        maxlevel = 1
        for tn in niter:
            if tn.is_regular():
                tn_level = swc.branch_order(tn)
                if tn_level >maxlevel:
                    maxlevel = tn_level
                if tn_level>level_limit:
                    print("too long")
                    maxlevel = level_limit+50
                    break
        return maxlevel

    def get_level_list(self):
        nodelist = []
        niter = iterators.LevelOrderIter(self.rawdata._root)
        maxlevel = self.maxbranchorder()
        self.maxlevel = maxlevel
        for i in range(maxlevel):
            list1 = []
            nodelist.append(list1)
        for tn in niter:
            if tn.is_regular():
                tn_level = swc.branch_order(tn)
                nodelist[tn_level-1].append(tn)
        self.nodelist = nodelist
        return nodelist

    def get_length_t(self):
        length_t = np.zeros([level_limit,level_limit])
        level = len(self.nodelist)
        for l in range(level):
            number = len(self.nodelist[l])
            for i in range(number):
                length_t[l][i] = self.nodelist[l][i].parent_distance()
        self.length_t = length_t

    def get_radius_t(self):
        radius_t = np.zeros([level_limit,level_limit])
        level = len(self.nodelist)
        for l in range(level):
            number = len(self.nodelist[l])
            for i in range(number):
                radius_t[l][i] = self.nodelist[l][i]._radius
        self.radius_t = radius_t

    def get_angle_t(self):
        angle_t = np.zeros([level_limit,level_limit])
        level = len(self.nodelist)
        for l in range(level):
            number = len(self.nodelist[l])
            for i in range(number):
                angle_t[l][i] = 0-swc.parent_angle(self.nodelist[l][i])
        self.angle_t = angle_t

    def get_type_t(self):
        type_t = np.zeros([level_limit,level_limit])
        level = len(self.nodelist)
        for l in range(level):
            number = len(self.nodelist[l])
            for i in range(number):
                type_t[l][i] = self.nodelist[l][i]._type
        self.type_t = type_t

    def get_plocation_t(self):
        plocation_t = np.zeros([level_limit,level_limit])
        level = len(self.nodelist)
        for l in range(level):
            number = len(self.nodelist[l])
            for i in range(number):
                plocation_t[l][i] = self.nodelist[l][i].parent._id
                if self.nodelist[l][i]._id == 1:
                    plocation_t[l][i] = 1
                else:
                    number_l = len(self.nodelist[l-1])
                    for m in range(number_l):
                        if plocation_t[l][i] == self.nodelist[l-1][m]._id:
                            plocation_t[l][i] = m+1
        self.plocation_t = plocation_t

    def save_length(self,path):
        np.savetext(path, self.length_t)

    def save_all(self,path):
        self.datall = np.zeros([level_limit,level_limit,4])
        self.datall[:,:,0] = self.length_t
        self.datall[:,:,1] = self.radius_t
        self.datall[:,:,2] = self.angle_t
        self.datall[:,:,3] = self.plocation_t
        scipy.misc.imsave(path,self.datall)
    def tran_file(self,pathdir):
        filelist = os.listdir(pathdir)
        for i in range(0,len(filelist)):
            path = os.path.join(pathdir,filelist[i])
            if os.path.isfile(path):
                print("%s"%path)
                self.tran_item(path,i)
        print("transfer file complete!")

    def tran_item(self,path,order):
        tree = swc.SwcTree()
        tree.load(path)
        self.load(tree)
        self.maxlevel = self.maxbranchorder()
        if self.maxlevel<=level_limit:
            self.get_level_list()
            self.get_length_t()
            self.get_radius_t()
            self.get_plocation_t()
            self.get_type_t()
            self.save_all('%s.png'%path[7:][:-4])
        else:
            print("too depth")
            #os.remove(path)

    def save(self,path):
        with open(path, 'w') as fp:
            for i in range(level_limit):
                for j in range(level_limit):
                    fp.write('%f ' % (self.angle_t[i][j]))
                fp.write('\n')
            fp.close()

    def cuta_file(self,pathdir,percent=0.05):
        filelist = os.listdir(pathdir)
        for i in range(0,len(filelist)):
            path = os.path.join(pathdir,filelist[i])
            if os.path.isfile(path):
                print("%s"%path)
                self.cuta_item(path,percent)

    def cuta_item(self,path,percent=0.05):
        tree = swc.SwcTree()
        tree.load(path)
        tree.delete_minpath(percent)
        if tree.cutcount != 0:
            tree.save('%scut%d.swc'%(path[5:][:-4],tree.cutcount))
    

    def maxbranchorder1(self):
        niter = iterators.LevelOrderIter(self.rawdata._root)
        maxlevel = 1
        for tn in niter:
            if tn.is_regular():
                tn_level = swc.branch_order(tn)
                if tn_level >maxlevel:
                    maxlevel = tn_level
        print(maxlevel)

    def tran1_item(self,path,order):
        tree = swc.SwcTree()
        tree.load(path)
        self.load(tree)
        self.maxlevel = self.maxbranchorder1()

    def tran1_file(self,pathdir):
        filelist = os.listdir(pathdir)
        for i in range(0,len(filelist)):
            path = os.path.join(pathdir,filelist[i])
            if os.path.isfile(path):
                print("%s"%path)
                self.tran1_item(path,i)
        print("transfer file complete!")






