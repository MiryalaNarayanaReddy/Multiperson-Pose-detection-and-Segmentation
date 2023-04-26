import matplotlib.pyplot as plt
im = plt.imread('results/images/000000.jpg')
w,h = im.shape[0],im.shape[1]
# plt.scatter([int(542.8700561523438),int(274.68609619140625)%h],[int(562.2120361328125)%w,int(255.60769653320312)%h],)
plt.imshow(im)
plt.show()
