from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


## 首先直接使用此方法
def read_tensorboard_data(tensorboard_path, name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    tag = ea.Tags()
    print(tag)
    key = ea.scalars.Keys()
    print(key)
    val = ea.scalars.Items(name)
    return val



def draw_plt(val, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name)
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('step')
    plt.ylabel(val_name)
    plt.show()


def getTwoDimensionListIndex(L,value):  
    """获得二维列表某个值的一维索引值 
    思想：先选出包含value值的一维列表，然后判断此一维列表在二维列表中的索引 
    """  
    data = [data for data in L if data[1]==value] #data=[(53, 1016.1)]  
    index = int(np.argwhere((L==data[0]).all(axis=1)))
    return index

# 另一种，只适用这一种的方法
def getindex(L, value):
    """
    一种只适用于读取tensorboard保存的v2版本的文件 因为step想当于index
    """
    data = [data for data in L if data[1]==value]
    index = data[0][0]
    return index


"""
本文方法只能读取一种v2版本的
在read_tensorboard_data的函数中通过ea.Tags()函数时可以看出
{'images': [], 'audio': [], 'histograms': [], 'scalars': [], 'distributions': [], 'tensors': ['evaluation_loss_vs_iterations', 'evaluation_Euclidean_vs_iterations', 'evaluation_cosine_similarity_vs_iterations', 'epoch_loss', 'epoch_Euclidean', 'epoch_cosine_similarity'], 
'graph': False, 'meta_graph': False, 'run_metadata': []}
如上面的字典中，如果读取到的数据时'scalars'为[], 而'tensors'不为空时，这种就会报错
read_tensorboard_data函数中的key为[]
"""


if __name__ == "__main__":
    tensorboard_train_path = './loop_noise5_cbamconcat_rms1_batchsize512/train/events.out.tfevents.1666014996.MS-IJHPFNLAGMLT.11164.17195.v2'
    
    tensorboard_val_path = './loop_noise5_cbamconcat_rms1_batchsize512/validation/events.out.tfevents.1666015011.MS-IJHPFNLAGMLT.11164.37325.v2'

    # 读取epcoh_loss, 因为模型是以val中的epoch_loss为指标保存的
    val_epoch_loss_name = 'epoch_loss'
    val_lo = read_tensorboard_data(tensorboard_val_path, val_epoch_loss_name) #读取tendorboard版本的文件的内容
    val_loss = [[i.step, i.value] for i in val_lo] #遍历val_lo的值，得到step、value的值
    val_loss = np.array(val_loss)
    val_loss_min = np.min(val_loss, axis=0)[1] #得到val_loss中的最小值
    print("val_epoch_loss:{}".format(val_loss_min))
    idx_min = getindex(val_loss, val_loss_min) #得到val_loss最小值对应的索引，即step的值，方便val_epoch_euclidean、train_loss、train_epoch_euclidean得到对应step的值
    print(idx_min)
    draw_plt(val_lo, val_epoch_loss_name)
    
    # 读取val中的epoch_Euclidean值，并根据val_epoch_loss中最小值的索引找到对应的epoch_Euclidean的值
    val_epoch_Euclidean_name = 'epoch_Euclidean'
    val_eucl = read_tensorboard_data(tensorboard_val_path, val_epoch_Euclidean_name)
    val_e_value = [i.value for i in val_eucl if i.step == idx_min]
    print("val_epoch_euclidean".format(val_e_value[0]))
    draw_plt(val_eucl, val_epoch_Euclidean_name)
    
    # 读取train中的epoch_loss值，并根据val_epoch_loss中最小值的索引找到对应的train_epoch_loss的值
    train_epoch_loss_name = 'epoch_loss'
    train_lo = read_tensorboard_data(tensorboard_train_path, train_epoch_loss_name)
    train_loss = [i.value for i in train_lo if i.step == idx_min]
    print("train_epoch_loss:{}".format(train_loss[0]))
    draw_plt(train_lo, train_epoch_loss_name)
    
    
    # 读取train中的epoch_Euclidean值，并根据val_epoch_loss中最小值的索引找到对应的train_epoch_Euclidean的值
    train_epoch_Euclidean_name = 'epoch_Euclidean'
    train_eucl = read_tensorboard_data(tensorboard_train_path, train_epoch_Euclidean_name)
    train_e_value = [i.value for i in train_eucl if i.step == idx_min]
    print("train_epoch_euclidean:{}".format(train_e_value[0]))
    draw_plt(train_eucl, train_epoch_Euclidean_name)