# experiment333
计算机视觉实验三
<font face="黑体" size=12><center>实验三、图像拼接</center></font>

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

## 6、拟合单应矩阵

- 首先定义单应矩阵的计算函数 `solve_homography()` ，如下所示：

```python

```

- 基于 RANSAC 算法，定义了 `fitHomoMat(matches_pos,nIter, th)` 函数，其基本思路如下：
  1. 首先将输入的匹配点对划分为源点集与目标点集，抽样计算初始单应矩阵  `𝐻` 。
  2. 将初始矩阵 `𝐻` 于所有源点进行投影变换，计算投影位置与实际目标点间的欧氏距离。若该距离小于设定阈值 `th` ，则认为该点对为内点（`inlier`）。
  3. 迭代（`nIter`）次比较各模型内点数量，选取内点最多的模型作为最优单应矩阵。

```python

```

# 三、实验总结与探索
1、本实验手动实现了图像拼接过程，并对不同的融合策略进行了横向对比；比较了手动实现方式、实验参考代码（作为baseline）、OpenCV API 这三种方式在不同场景下的拼接效果。
