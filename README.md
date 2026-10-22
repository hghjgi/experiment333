# experiment333
<font face="宋体" size=4>
<b>
<!--文件头 第一行 需要改动!!!!!!!!-->
<div>
    <span style="float: left">学生姓名: 彭艺琼</span>
    <span style="float: right">指导老师: 胡政伟</span>
</div>
<center>学号: 202410315019 </center>
<br>

# 一、实验目的

通过SIFT算法提取并匹配图像特征点，结合RANSAC算法实现图像配准与拼接，对比多种图像融合策略及图像拼接不同实现方式的差异。

# 二、实验内容

## 1、图像拼接一般步骤
  1. 特征匹配，对读入的两张图片进行特征点匹配。
  2. 筛选特征点对，计算单应矩阵，将两张图片放到同一个平面坐标系下。
  3. 采取相应的图像融合策略，对重叠区域进行平滑，消除拼接边界。

## 2、输入图像
- 拼接使用的图像为啊a.jpg和b.jpg，展示如下。
- a.jpg <img width="1256" height="1231" alt="8c3a59df8a356de5464fb7c53b64ac7" src="https://github.com/user-attachments/assets/dc1930ba-ee39-4384-96ef-5a9ae4004a4c" />
- b.jpg <img width="977" height="1231" alt="e16e98b639af1b6a461d2c22862e01d" src="https://github.com/user-attachments/assets/cd420bdc-3b6d-4240-9ed4-a8b2fed3fe7f" />

## 3、获取特征点
- 初始化SIFT检测器并提取特征
```python
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)  # 图像a的关键点和描述符
kp_b, des_b = sift.detectAndCompute(gray_b, None)  # 图像b的关键点和描述
```
## 4、特征匹配
- 使用FLANN匹配器进行特征匹配
```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)  # k=2表示每个特征点返回2个最佳匹配

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
   if m.distance < 0.7 * n.distance:  # 比率阈值通常取0.7-0.8
       good_matches.append(m)

matched_keypoints_img = cv2.drawMatches(
   img_a, kp_a, img_b, kp_b, good_matches,
   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像b的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像a的关键点
```

## 5、RANSAC算法
> - RANSAC算法的基本假设是样本中包含正确数据(inliers，可以被模型描述的数据)，也包含异常数据(outliers，偏离正常范围很远、无法适应数学模型的数据)，即数据集中含有噪声。
> - 由直线的知识点可知，两点可以确定一条直线，所以可以随机的在数据点集中选择两点，从而确定一条直线。然后通过设置给定的阈值，计算在直线两旁的符合阈值范围的点，统计点的个数inliers。inliers最多的点集所在的直线，就是我们要选取的最佳直线。
```python
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

## 6、图像尺寸确定
- 获取输入图像尺寸，计算图像b变换后的四个角点坐标，确定拼接后图像的最终尺寸
```python
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], 
[0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
```
## 7、图像拼接
- 创建平移矩阵，确保所有像素都在可见区域内
- 对图像b进行透视变换和平移
- 将图像a复制到拼接结果的对应位置，显示匹配关键点和拼接结果，保存拼接结果
```python
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

stitched_img = cv2.warpPerspective(
img_b,
translation_matrix @ H, 
(x_max - x_min, y_max - y_min)
)

stitched_img[-y_min : h_a - y_min, -x_min : w_a - x_min] = img_a

cv2.imshow('Matched SIFT Keypoints', matched_keypoints_img)  # 显示匹配的关键点
cv2.imshow('Stitched Result', stitched_img)                  # 显示拼接后的图像
cv2.waitKey(0)  # 等待用户按键（按任意键关闭窗口）
cv2.destroyAllWindows()  # 关闭所有显示窗口
```
-最终得到拼接图像如下：
<img width="944" height="739" alt="image" src="https://github.com/user-attachments/assets/14b8b71b-50cd-4bd2-9aba-1d38ab2413a9" />

# 三、实验总结与探索
- 本次实验利用SIFT实现图像全景拼接，完成特征提取、FLANN匹配、RANSAC求解单应矩阵与透视变换，熟悉全景拼接完整流程。
- Lowe比率控制误匹配数量，RANSAC进一步筛选有效内点，二者共同保障图像对齐的准确性。
- 图像重叠度、纹理情况会直接影响拼接效果，后续可引入图像融合消除拼接缝隙。

# 四、实验中遇到的问题与解决方案
- 问题：初始特征匹配存在大量误匹配点，导致图像拼接错乱、拼接效果差。
- 解决方案：通过调低Lowe比率阈值筛选优质匹配点，同时利用RANSAC算法剔除错误匹配外点，精准求解单应矩阵，有效修正图像透视位置，最终实现正常拼接。
- 错误图像如下：
- <img width="1280" height="800" alt="25fa741cf4c71c8ab84ba2e4a861e82" src="https://github.com/user-attachments/assets/804aafa3-c1a5-47f6-a5a8-f3e9aff02f6c" />

