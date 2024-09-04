import numpy as np
import scipy.integrate as spi
from scipy.optimize import root_scalar
import skimage
from skimage.morphology import dilation, cube, skeletonize

class Slices:
    def __init__(self, image, mandible):
        self.slice_size = (200, 200)
        self.n_slices = 200
        self.image = image
        self.mandible = mandible
        self.dilated_roi = dilation(self.mandible, cube(21)) # 截取下颌骨区域，适当膨胀作为ROI
        self.image[self.dilated_roi == 0] = 0

        self.skeleton_points = None # 骨架点坐标
        self.p = None # 牙弓曲线拟合多项式
        self.centers = None # 采样点坐标
        self.normals = None # 采样点切向量
        self.slices = None # 切片
        self.aug_dict = None # 坐标变换矩阵
        
    def process(self):
        self.skeleton_points, self.p = self.get_arch()
        self.centers, self.normals = self.get_sample_points()
        self.slices, self.aug_dict = self.get_slices()
        return self.slices, self.aug_dict
    
    def detect(self):
        """检测切片中的高置信度点"""


    def get_arch(self):
        """提取骨架并拟合出牙弓曲线"""
        """
        Returns:
            skeleton_points: 骨架点坐标
            p: 牙弓曲线拟合多项式
        """
        flatten_image = np.sum(self.mandible, axis=0) > 0 # 将下颌骨纵向展平
        flatten_image = skimage.morphology.opening(flatten_image, skimage.morphology.square(19))
        image_skeleton = skeletonize(flatten_image.astype(np.uint8)) # 提取骨架
        rows, cols = np.where(image_skeleton == 1) # 骨架坐标
        skeleton_points = np.column_stack((cols, rows)) # 骨架点列表
        # 骨架点排序
        skeleton_points = np.array(sorted(skeleton_points, key=lambda x: x[0])) # 按x坐标排序

        # 等间隔取点用于拟合牙弓曲线
        # control_points_inx = np.linspace(0, len(skeleton_points)-1, 40, dtype=int)
        # control_points = skeleton_points[control_points_inx, :] # 控制点坐标
        # 用二次多项式进行拟合
        eff = np.polyfit(skeleton_points[:, 0], skeleton_points[:, 1], 2)
        p = np.poly1d(eff) # 牙弓曲线表达式
        return skeleton_points, p

    def get_sample_points(self):
        # 采样点坐标
        x_lim = [self.skeleton_points[:, 0].min(), self.skeleton_points[:,0].max()]
        x = np.linspace(x_lim[0], x_lim[1], self.n_slices)
        # y = p(x)
        # x_lim = [skeleton_points[0][0], skeleton_points[-1][0]]
        centers = self.resample(x_lim)  # 重采样
        # 采样点切向量
        p_deriv = np.polyder(self.p)
        normals = np.column_stack((np.ones_like(x), p_deriv(x))) # 法向量
        normals = normals / np.linalg.norm(normals, axis=1)[:, None] # Normalization
        return centers, normals

    def get_slices(self):
        # 切片数组
        image_slice_list = []
        # 坐标变换矩阵
        A_aug_dict = {}
        counter = 0 # 计数器
        mandible_min = np.argwhere(self.mandible == 1)[0][0] # 下颌骨最低点
        for center, normal2d in zip(self.centers, self.normals): # 遍历采样点, 计算每个点对应的坐标变换矩阵
            # 创建切片
            image_slice = np.zeros(self.slice_size)
            # 切片二维坐标
            i, j = np.indices(image_slice.shape)
            i = i - self.slice_size[0] // 2  # 坐标原点不在切片的角落，而是在正下方中心，在还原的时候要注意对横坐标进行平移
            # j = j + mandible_min
            # 计算平移向量
            translation = np.array([[mandible_min], [center[1]], [center[0]]])
            # 计算旋转矩阵
            normal = [0, normal2d[1], normal2d[0]]
            base2 = np.array([1, 0, 0])
            base1 = self.schmidt(normal, base2) # 三个基向量
            A = np.array([normal, base1, base2]).T # 旋转矩阵
            # 坐标转换矩阵
            A_aug = np.hstack((A, translation))
            A_aug = np.vstack((A_aug, np.array([0, 0, 0, 1])))
            
            # 切片三维增广坐标
            x_aug = np.stack([np.zeros_like(i), i, j, np.ones_like(i)], axis=-1)
            # 计算切片中各点在原坐标系下的坐标
            x_aug_original = np.squeeze(A_aug @ x_aug[..., None]).astype(int)
            # 保留在原图像范围内的点
            valid = np.logical_and.reduce((x_aug_original[..., 0] >= 0, x_aug_original[..., 0] < self.image.shape[0],
                                        x_aug_original[..., 1] >= 0, x_aug_original[..., 1] < self.image.shape[1],
                                        x_aug_original[..., 2] >= 0, x_aug_original[..., 2] < self.image.shape[2]))
            

            image_slice[valid] = self.image[x_aug_original[valid, 0], x_aug_original[valid, 1], x_aug_original[valid, 2]]
            image_slice_list.append(image_slice)
            A_aug_dict["{:0>3d}".format(counter)] = A_aug.tolist()
            # A_aug_list.append({"{:0>3d}".format(counter): A_aug.tolist()})
            counter += 1
        
        return image_slice_list, A_aug_dict

    def resample(self, x_lim):
        """Resample a curve defined by a polynomial function to get evenly spaced points"""
        """
        Args:
            x_lim: 采样的x坐标范围
            polyfunc: 多项式函数
            num_pts: 采样点数
        """
        x1, x2 = x_lim
        my_poly = self.p
        num_points = self.n_slices
        def curve_length(x):
            return spi.quad(lambda t: np.sqrt(1 + my_poly(t)**2), x1, x)[0] # 求定积分
        total_curve_length = curve_length(x2)

        # 计算等间距点之间的理论距离
        desired_spacing = total_curve_length / (num_points - 1)

        # 定义函数来查找使得两点间距离等于 desired_spacing 的 x 坐标
        def find_x_for_distance(d):
            result = root_scalar(lambda x: curve_length(x) - d, bracket=[x1, x2], method='bisect')
            return result.root

        # 生成均匀间隔的点
        interpolated_x = [x1]
        for i in range(1, num_points - 1):
            desired_length = i * desired_spacing
            interpolated_x.append(find_x_for_distance(desired_length))
        interpolated_x.append(x2)

        # 计算每个点对应的 y 值
        interpolated_y = [my_poly(x) for x in interpolated_x]

        # 打印插值点
        interpolated_points = np.column_stack((interpolated_x, interpolated_y))

        return interpolated_points
    
    def schmidt(sels, base1, base2):
        '''施密特正交单位化'''
        base1 = np.array(base1)
        base2 = np.array(base2)
        base1 = base1 / np.linalg.norm(base1)
        base2 = base2 / np.linalg.norm(base2)
        base3 = np.cross(base1, base2)
        base3 = base3 / np.linalg.norm(base3)
        return base3