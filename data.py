import numpy as np

a = np.array([[8, 10, 5], [1, 2, 3], [3, 4, 10]])
a_min = np.min(a, axis=0)[1] #得到a第二维中最小的值是 2
b = [data for data in a if data[1]==a_min][0] #得到最小的值所处的值，如[1, 2]
print(b)
print(a == b)
print((a == b).all(axis=1)) 
print((a == b).all(axis=0)) 
pos = np.where((a==b).all(axis=1))
print(a[pos])
print(np.argwhere((a==b).all(axis=1)))
print(int(np.argwhere((a==b).all(axis=1))))

# summaries = tf.compat.v1.train.summary_iterator(tensorboard_val_path)
# epoch_loss = []

# for e in summaries:
#     for v in e.summary.value:
#         if v.tag == 'epoch_loss':
#             epoch_loss.append(tf.make_ndarray(v.tensor))
# print(len(epoch_loss))  