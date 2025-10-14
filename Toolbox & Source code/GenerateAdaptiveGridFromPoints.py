import arcpy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

# 计算数据集的范围
def get_bound(points):
    min_lon, min_lat = np.min(points, axis=0)
    max_lon, max_lat = np.max(points, axis=0)
    # 扩展5%的边界范围以避免边缘点落在边界上
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    min_lon -= lon_range * 0.05
    max_lon += lon_range * 0.05
    min_lat -= lat_range * 0.05
    max_lat += lat_range * 0.05
    return min_lon, min_lat, max_lon, max_lat

class QuadTreeNode:
    def __init__(self, x_min, y_min, x_max, y_max, depth=0):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.depth = depth
        self.children = None
        self.point_indices = []
    def is_leaf(self):
        return self.children is None
    
    def split(self, threshold=10):
        
        if len(self.point_indices) <= threshold:
            return
        # 计算分割点
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2


        # 创建四个子节点
        self.children = [
            QuadTreeNode(self.x_min, y_mid, x_mid, self.y_max, self.depth + 1),  # 左上
            QuadTreeNode(x_mid, y_mid, self.x_max, self.y_max, self.depth + 1),  # 右上
            QuadTreeNode(self.x_min, self.y_min, x_mid, y_mid, self.depth + 1),  # 左下
            QuadTreeNode(x_mid, self.y_min, self.x_max, y_mid, self.depth + 1)  # 右下
        ]
        
        # 将点分配到子节点
        for idx in self.point_indices:
            
            point = points[idx]
            for child in self.children:
               
                if child.contains(point):
 
                    child.point_indices.append(idx)

                    break

        # 清空当前节点的点
        self.point_indices = []
        # 递归分割子节点
        for child in self.children:
            child.split(threshold)
    def contains(self, point):
        return (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max)
    def get_all_cells(self):
        cells = []
        if self.is_leaf():
            cells.append(self)
        else:
            for child in self.children:
                cells.extend(child.get_all_cells())
        return cells


# 将网格包含的点编号信息打印到控制台
def print_cell(cells):

    for i, cell in enumerate(cells, 1):
        if len(cell.point_indices) > 0:
            point_ids = ",".join(map(str, cell.point_indices))

def write_cell_fc(output_grid_fc, input_fc_sr, cells):

    # 设置工作空间
    arcpy.env.overwriteOutput = True
        
    # 创建临时要素类名称
    temp_origin_fc = "in_memory\\temp_origin"

    fc_path = os.path.dirname(temp_origin_fc)
    fc_name = os.path.basename(temp_origin_fc)

    # 创建要素类
    arcpy.CreateFeatureclass_management(
        out_path=fc_path,
        out_name=fc_name,
        geometry_type="POINT",
        spatial_reference=input_fc_sr
    )

    # 添加字段
    arcpy.AddField_management(temp_origin_fc, "grid_id", "LONG")
    arcpy.AddField_management(temp_origin_fc, "x", "DOUBLE")
    arcpy.AddField_management(temp_origin_fc, "y", "DOUBLE")

    # 准备要插入的数据

    with arcpy.da.InsertCursor(temp_origin_fc, ["grid_id", "x", "y", "SHAPE@XY"]) as cursor:
        for i, cell in enumerate(cells, 1):
            grid_id = i
            x = cell.x_min
            y = cell.y_max
            row = (grid_id, x, y, (x, y))
            cursor.insertRow(row)

            grid_id = i
            x = cell.x_max
            y = cell.y_max
            row = (grid_id, x, y, (x, y))
            cursor.insertRow(row)

            grid_id = i
            x = cell.x_max
            y = cell.y_min
            row = (grid_id, x, y, (x, y))
            cursor.insertRow(row)

            grid_id = i
            x = cell.x_min
            y = cell.y_min
            row = (grid_id, x, y, (x, y))
            cursor.insertRow(row)

    arcpy.AddMessage(f"成功创建点要素类 {fc_name} 并插入了 {i*4} 条记录")

    # 创建临时要素类名称
    temp_line_fc = "in_memory\\temp_line"

    fc_path = os.path.dirname(temp_origin_fc)
    fc_name = os.path.basename(temp_origin_fc)

    arcpy.AddMessage("开始基于点创建闭合线要素类...")

    arcpy.management.PointsToLine(
        Input_Features=temp_origin_fc,
        Output_Feature_Class= temp_line_fc,
        Line_Field="grid_id",
        Sort_Field=None,
        Close_Line="CLOSE",
        Line_Construction_Method="CONTINUOUS",
        Attribute_Source="NONE",
        Transfer_Fields=None
    )

    arcpy.AddMessage("基于点创建闭合线要素类完成！")

    arcpy.management.FeatureToPolygon(
        in_features = temp_line_fc,
        out_feature_class = output_grid_fc,
        cluster_tolerance = None,
        attributes = "ATTRIBUTES",
        label_features = None
    )


def script_tool(input_points_fc, extent_fc, minimum_points, output_grid_fc):

    """Script code goes below"""

    # 获取输入要素类的空间参考
    desc = arcpy.Describe(input_points_fc)
    input_fc_sr = desc.spatialReference

    arcpy.AddMessage(f"输入要素的空间参考为：{input_fc_sr.name}")

    # 获取要素类的范围(extent)
    extent = arcpy.Describe(extent_fc).extent

    # 获取范围的各个坐标值
    xmin = extent.XMin
    ymin = extent.YMin
    xmax = extent.XMax
    ymax = extent.YMax

    root_ex = QuadTreeNode(xmin, ymin, xmax, ymax)

    # 将点添加到四叉树
    for i in range(len(points)):
        root_ex.point_indices.append(i)

    # 分割四叉树, threshold表示一个网格中如果超过4个点就会被细分
    root_ex.split(int(minimum_points))

    # 获取所有叶子单元格
    cells = root_ex.get_all_cells()

    arcpy.AddMessage("写入划分方案到结果要素类中...")
    write_cell_fc(output_grid_fc, input_fc_sr, cells)
    arcpy.AddMessage("划分方案写入结果要素类完成！")
        
    return
if __name__ == "__main__":

    input_points_fc = arcpy.GetParameterAsText(0)
    extent_fc = arcpy.GetParameter(1)
    minimum_points = arcpy.GetParameter(2)
    output_grid_fc = arcpy.GetParameterAsText(3)

    points = []

    # 读取点要素类中的每个点
    with arcpy.da.SearchCursor(input_points_fc, ["SHAPE@XY"]) as cursor:
        for row in cursor:
            # 获取XY坐标对
            x, y = row[0]
            # 将坐标对添加到列表
            points.append([x, y])

    points = np.array(points)

    script_tool(input_points_fc, extent_fc, minimum_points, output_grid_fc)
