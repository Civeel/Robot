import cv2  # 默认读取格式为BGR，建议用opencv函数
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('f:/test.jpg')  # 路径
# print(img)
# cv2.imshow('image', img)
# cv2.waitKey(0)  # 等待时间，0为按下任意键关闭
# cv2.destroyAllWindows()

'''
    读取图片
'''


# (传入图片)定义为函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv_show('image', img)
# print(img.shape)  # 宽，长，3为RGB类型

# img_gray = cv2.imread('f:/test.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
# print(img_gray)
# print(img_gray.shape)  # 灰度图只有两个数据
# cv_show('img_gray', img_gray)

# 图像保存
# cv2.imwrite('f:/mytest.png', img_gray)

# print(img.size)  # 图像的像素值
# print(img.dtype)  # 图像的类型


'''
    读取视频
'''
# vc = cv2.VideoCapture('f:/烟花.mov')  # 参数为0，则打开摄像头
# if vc.isOpened():
#     open, frame = vc.read()  # 读取第一帧
# else:
#     open = False
#
# while open:
#     ret, frame = vc.read()  # 读取第一帧
#     if frame is None:
#         break
#     if ret:
#         # 将这一帧转换为灰度图
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.flip(frame, 0)  # 调整是否镜像（1为镜像）
#         cv2.imshow('result', gray)
#         if cv2.waitKey(1) & 0xFF == 27:  # 数字越大越慢（每一帧之间的间隔时长）,按esc退出
#             break
# vc.release()
# cv2.destroyAllWindows()


'''
    截取部分图像数据
    颜色通道提取
'''
img = cv2.imread('f:/test.jpg')
# test = img[100:500, 100:500]  # 高度，长度
# cv_show('test', test)

b, g, r = cv2.split(img)  # 三个颜色通道分别提取，顺序一定是BGR
# print(b.shape)  # 三者的shape是不变的
img = cv2.merge((b, g, r))  # BGR重组
# print(img.shape)

# 只保留B、G、R
cur_img = img.copy()
cur_img[:, :, 0] = 0  # B = 0
cur_img[:, :, 1] = 0  # G = 0
# cur_img[:, :, 2] = 200  # R = 0
# cv_show('R', cur_img)

'''
边界填充
'''
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)  # 上下左右填充的宽度

replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)  # 按照borderType的方式填充
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT)

import matplotlib.pyplot as plt
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')  # 复制最边缘像素
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')  # 反射法
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT101')  # 101反射法，以最边缘像素为轴反射
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')  # 外包装法
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')  # 常数值填充
plt.show()
