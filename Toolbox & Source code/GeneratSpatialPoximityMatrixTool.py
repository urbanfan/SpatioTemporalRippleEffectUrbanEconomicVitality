import arcpy
import os
import csv


def create_neighbor_matrix_using_tools(input_fc, node_id, output_csv_file):
    
    arcpy.AddMessage("使用空间权重矩阵工具开始生成邻近关系...")
    
    # 获取默认工作空间和输出目录
    default_ws = arcpy.env.workspace
    pre_dir = os.path.dirname(default_ws)
    
    # 设置临时文件和最终输出路径
    temp_swm_file = os.path.join(pre_dir, "temp_weights.swm")

    temp_table = os.path.join(pre_dir, "temp_neighbor_table")
    neighbor_matrix_csv = output_csv_file
    
    # 步骤1: 生成空间权重矩阵 (.swm文件)
    arcpy.AddMessage("正在生成空间权重矩阵...")

    arcpy.stats.GenerateSpatialWeightsMatrix(
        Input_Feature_Class=input_fc,
        Unique_ID_Field=node_id,
        Output_Spatial_Weights_Matrix_File=temp_swm_file,
        Conceptualization_of_Spatial_Relationships="CONTIGUITY_EDGES_CORNERS",
        Distance_Method="EUCLIDEAN",
        Exponent=1,
        Threshold_Distance=None,
        Number_of_Neighbors=0,
        Row_Standardization="NO_STANDARDIZATION"
    )

    arcpy.AddMessage("空间权重矩阵生成完成！")
    
    # 步骤2: 将空间权重矩阵转换为表格
    arcpy.AddMessage("正在将空间权重矩阵转换为表格...")

    arcpy.stats.ConvertSpatialWeightsMatrixtoTable(
        Input_Spatial_Weights_Matrix_File=temp_swm_file,
        Output_Table=temp_table
    )

    arcpy.AddMessage("空间权重矩阵转换为表格完成")
    
    # 步骤3: 从表格中提取邻居关系并保存为CSV
    arcpy.AddMessage("正在将表格数据转换为CSV格式...")
    neighbors = {}
    
    # 从表格中读取邻居关系
    with arcpy.da.SearchCursor(temp_table, [node_id, "NID"]) as cursor:
        for row in cursor:
            source_id = row[0]
            neighbor_id = row[1]
            
            if source_id not in neighbors:
                neighbors[source_id] = set()
                
            neighbors[source_id].add(neighbor_id)
    
    # 将结果写入CSV文件
    with open(neighbor_matrix_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for region_id, neighbor_ids in neighbors.items():
            row = [region_id] + list(neighbor_ids)
            writer.writerow(row)
    
    arcpy.AddMessage(f"邻近关系已保存到: {neighbor_matrix_csv}")
    
    # 清理临时文件
    if arcpy.Exists(temp_table):
        arcpy.Delete_management(temp_table)
    if os.path.exists(temp_swm_file):
        os.remove(temp_swm_file)
    
    return neighbor_matrix_csv

def script_tool(input_areal_featurelayer, area_id_field, output_csv_file):
    """Script code goes below"""


    arcpy.AddMessage("开始遍历每个单元寻找其邻近单元...")

    neighbor_matrix_csv = create_neighbor_matrix_using_tools(input_areal_featurelayer, area_id_field, output_csv_file)

    arcpy.AddMessage("遍历每个单元寻找其邻近单元完成!")
    arcpy.AddMessage("----------------------------------")

    return


if __name__ == "__main__":

    input_areal_featurelayer = arcpy.GetParameterAsText(0)
    area_id_field = arcpy.GetParameterAsText(1)
    output_csv_file = arcpy.GetParameterAsText(2)

    script_tool(input_areal_featurelayer, area_id_field, output_csv_file)