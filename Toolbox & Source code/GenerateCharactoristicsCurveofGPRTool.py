import arcpy
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Callable, Optional
def find_most_representative_timeseries(
    timeseries_dict: Dict[Union[str, int], List[float]],
    distance_metric: str = 'euclidean',
    normalize: bool = True,
    plot_result: bool = False,
    custom_distance_func: Optional[Callable] = None
) -> Tuple[Union[str, int], List[float], float]:
    """
    找出与其他所有时间序列差异性最小的时间序列
    
    参数:
    timeseries_dict (Dict): 时间序列字典，key为序列编号，value为时序值列表
    distance_metric (str): 距离计算方法，可选值包括:
                          'euclidean' - 欧氏距离
                          'manhattan' - 曼哈顿距离
                          'correlation' - 相关性距离 (1 - 相关系数)
                          'pearson' - 皮尔逊相关距离
                          'spearman' - 斯皮尔曼相关距离
    normalize (bool): 是否对时间序列进行归一化处理
    plot_result (bool): 是否绘制结果图表
    custom_distance_func (Callable): 自定义距离计算函数，当distance_metric='custom'时使用
    
    返回:
    Tuple: 包含三个元素的元组:
           1. 最具代表性的时间序列编号
           2. 最具代表性的时间序列值列表
           3. 平均差异度分数
    """
    # 检查输入是否为空
    if not timeseries_dict:
        raise ValueError("输入的时间序列字典为空")
    
    # 检查所有时间序列长度是否一致
    lengths = [len(ts) for ts in timeseries_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError("所有时间序列的长度必须一致")
    
    # 将字典转换为数组形式，行为时间序列，列为时间点
    ts_ids = list(timeseries_dict.keys())
    ts_array = np.array([timeseries_dict[key] for key in ts_ids])
    
    # 如果需要归一化
    if normalize:
        # 减去均值并除以标准差 (Z-score标准化)
        ts_mean = np.mean(ts_array, axis=1, keepdims=True)
        ts_std = np.std(ts_array, axis=1, keepdims=True)
        # 避免除以0
        ts_std = np.where(ts_std == 0, 1, ts_std)
        ts_normalized = (ts_array - ts_mean) / ts_std
        
        # 使用归一化后的数据
        ts_array = ts_normalized
    
    # 根据指定的距离度量计算距离矩阵
    distance_matrix = None
    
    if distance_metric == 'euclidean':
        # 欧氏距离
        distance_matrix = squareform(pdist(ts_array, metric='euclidean'))
    
    elif distance_metric == 'manhattan':
        # 曼哈顿距离
        distance_matrix = squareform(pdist(ts_array, metric='cityblock'))
    
    elif distance_metric == 'correlation':
        # 相关性距离 (1 - 相关系数)
        distance_matrix = squareform(pdist(ts_array, metric='correlation'))
    
    elif distance_metric == 'pearson':
        # 计算皮尔逊相关系数距离
        n_series = len(ts_ids)
        distance_matrix = np.zeros((n_series, n_series))
        
        for i in range(n_series):
            for j in range(i+1, n_series):
                corr, _ = pearsonr(ts_array[i], ts_array[j])
                # 距离 = 1 - 相关系数的绝对值
                distance = 1 - abs(corr)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    elif distance_metric == 'spearman':
        # 计算斯皮尔曼相关系数距离
        n_series = len(ts_ids)
        distance_matrix = np.zeros((n_series, n_series))
        
        for i in range(n_series):
            for j in range(i+1, n_series):
                corr, _ = spearmanr(ts_array[i], ts_array[j])
                # 距离 = 1 - 相关系数的绝对值
                distance = 1 - abs(corr)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    elif distance_metric == 'custom' and custom_distance_func is not None:
        # 使用自定义距离函数
        n_series = len(ts_ids)
        distance_matrix = np.zeros((n_series, n_series))
        
        for i in range(n_series):
            for j in range(i+1, n_series):
                distance = custom_distance_func(ts_array[i], ts_array[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    else:
        raise ValueError(f"不支持的距离度量方法: {distance_metric}")
    
    # 计算每个时间序列与其他所有时间序列的平均距离
    mean_distances = np.zeros(len(ts_ids))
    for i in range(len(ts_ids)):
        # 排除自身与自身的距离(值为0)
        other_distances = np.concatenate([distance_matrix[i, :i], distance_matrix[i, i+1:]])
        mean_distances[i] = np.mean(other_distances)
    
    # 找到具有最小平均距离的时间序列
    min_index = np.argmin(mean_distances)
    min_distance = mean_distances[min_index]
    most_representative_id = ts_ids[min_index]
    most_representative_ts = timeseries_dict[most_representative_id]
    
    # 打印结果
    print(f"最具代表性的时间序列编号: {most_representative_id}")
    print(f"平均差异度分数: {min_distance:.4f}")
    
    # 如果需要绘图
    if plot_result:
        plt.figure(figsize=(12, 8))
        
        # 绘制所有时间序列
        for i, (ts_id, ts_values) in enumerate(timeseries_dict.items()):
            if ts_id == most_representative_id:
                # 突出显示最具代表性的时间序列
                plt.plot(ts_values, 'r-', linewidth=3, label=f"{ts_id} (最具代表性)")
            else:
                plt.plot(ts_values, 'b-', alpha=0.3, label=f"{ts_id}" if i < 5 else "")
        
        plt.title(f"时间序列比较 (使用{distance_metric}距离)")
        plt.xlabel("时间点")
        plt.ylabel("值" + (" (已归一化)" if normalize else ""))
        
        # 只显示部分图例以避免过度拥挤
        handles, labels = plt.gca().get_legend_handles_labels()
        # 选择显示最具代表性的和前几个其他序列
        selected_handles = [h for i, h in enumerate(handles) if i < 5 or labels[i].endswith("(最具代表性)")]
        selected_labels = [l for i, l in enumerate(labels) if i < 5 or l.endswith("(最具代表性)")]
        plt.legend(selected_handles, selected_labels, loc='best')
        
        plt.grid(True)
        
        # 绘制平均距离条形图
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(ts_ids)), mean_distances)
        plt.xticks(range(len(ts_ids)), ts_ids, rotation=45)
        plt.xlabel("时间序列编号")
        plt.ylabel("平均距离")
        plt.title("每个时间序列与其他序列的平均距离")
        plt.grid(True, axis='y')
        
        # 标记最小距离
        plt.plot(min_index, min_distance, 'ro', markersize=10)
        plt.annotate(f"最小: {min_distance:.4f}",
                    xy=(min_index, min_distance),
                    xytext=(min_index + 0.5, min_distance + 0.1 * np.max(mean_distances)),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        plt.show()
    
    return most_representative_id, most_representative_ts, min_distance
def add_single_row(table_path, subregion_id, dt, val):
    """
    向表中添加单行数据
    
    参数:
    table_path (str): 表的完整路径
    subregion_id (int): 子区域ID
    dt (datetime): 日期时间
    val (float): 数值
    
    返回:
    bool: 是否添加成功
    """
    try:
        # 检查表是否存在
        if not arcpy.Exists(table_path):
            print(f"错误: 表不存在 - {table_path}")
            return False
        
        print(f"向表 {table_path} 添加单行数据...")
        
        # 使用InsertCursor添加行
        with arcpy.da.InsertCursor(table_path, ["subregion_id", "dt", "val"]) as cursor:
            cursor.insertRow([subregion_id, dt, val])
        
        print(f"成功添加行: subregion_id={subregion_id}, dt={dt}, val={val}")
        return True
        
    except arcpy.ExecuteError:
        print(f"添加行时出错(ArcPy): {arcpy.GetMessages(2)}")
        return False
    except Exception as e:
        print(f"添加行时出错: {str(e)}")
        return False
def extract_sorted_dt_list(df, subregion_id_field, region_code_field, dt_field):
    """
    从DataFrame中提取第一个subregion_id值和第一个region_code值对应的所有行的dt值，
    并返回排序后的列表
    
    参数:
    df (pd.DataFrame): 包含所需字段的DataFrame
    subregion_id_field (str): 子区域ID字段名，默认为'subregion_id'
    region_code_field (str): 区域代码字段名，默认为'region_code'
    dt_field (str): 日期时间字段名，默认为'dt'
    
    返回:
    list: 排序后的dt值列表
    """
    try:
        # 检查输入的DataFrame是否为空
        if df.empty:
            print("错误: 输入的DataFrame为空")
            return []
        
        # 检查字段是否存在
        required_fields = [subregion_id_field, region_code_field, dt_field]
        for field in required_fields:
            if field not in df.columns:
                arcpy.AddMessage(f"错误: DataFrame中不存在字段 '{field}'")
                return []
        
        # 获取第一个subregion_id值
        first_subregion_id = df[subregion_id_field].iloc[0]
        arcpy.AddMessage(f"第一个subregion_id值: {first_subregion_id}")
        
        # 筛选该subregion_id的所有行
        subregion_df = df[df[subregion_id_field] == first_subregion_id]
        
        # 获取筛选后DataFrame的第一个region_code值
        if subregion_df.empty:
            arcpy.AddMessage(f"警告: 没有找到subregion_id为{first_subregion_id}的行")
            return []
        
        first_region_code = subregion_df[region_code_field].iloc[0]
        arcpy.AddMessage(f"第一个region_code值: {first_region_code}")
        
        # 筛选满足两个条件的所有行
        filtered_df = df[(df[subregion_id_field] == first_subregion_id) & 
                          (df[region_code_field] == first_region_code)]
        
        if filtered_df.empty:
            arcpy.AddMessage(f"警告: 没有找到同时满足subregion_id为{first_subregion_id}且region_code为{first_region_code}的行")
            return []
        
        # 获取dt值并排序
        dt_values = filtered_df[dt_field].tolist()
        
        # 排序（确保日期时间类型能正确排序）
        # 如果dt_field已经是日期时间类型，可以直接排序
        # 如果是字符串类型，尝试转换为日期时间再排序
        if isinstance(dt_values[0], (pd.Timestamp, np.datetime64)):
            sorted_dt_values = sorted(dt_values)
        else:
            try:
                # 尝试转换为日期时间后排序
                sorted_dt_values = sorted([pd.to_datetime(dt) for dt in dt_values])
            except:
                # 如果转换失败，按字符串排序
                arcpy.AddMessage("警告: dt值不是日期时间类型，将按字符串排序")
                sorted_dt_values = sorted(dt_values)
        
        arcpy.AddMessage(f"找到 {len(sorted_dt_values)} 个符合条件的dt值")
        
        return sorted_dt_values
        
    except Exception as e:
        arcpy.AddMessage(f"发生错误: {str(e)}")
        return []
def create_table_with_fields(table_path):
    """
    在指定路径创建一个包含subregion_id, dt, val三个字段的表
    字段类型分别为LONG, DATE, DOUBLE
    
    参数:
    table_path (str): 表的完整路径，包括文件夹或地理数据库路径和表名
    """
    try:
        # 分离路径和表名
        workspace, table_name = os.path.split(table_path)
        
        # 如果表名包含扩展名(如.dbf)，去除扩展名
        table_name = os.path.splitext(table_name)[0]
        
        print(f"工作空间: {workspace}")
        print(f"表名称: {table_name}")
        
        # 检查工作空间是否存在
        if not arcpy.Exists(workspace):
            print(f"错误: 工作空间不存在 - {workspace}")
            return None
        
        # 检查表是否已存在，如果存在则删除
        if arcpy.Exists(os.path.join(workspace, table_name)):
            arcpy.Delete_management(os.path.join(workspace, table_name))
            print(f"已删除现有表: {table_name}")
        
        # 创建新表
        print(f"正在创建表: {table_name}...")
        arcpy.CreateTable_management(workspace, table_name)
        
        # 定义字段列表
        fields = [
            ("subregion_id", "LONG", "", "子区域ID"),
            ("dt", "DATE", "", "日期时间"),
            ("val", "DOUBLE", "", "数值")
        ]
        
        # 添加字段
        print("正在添加字段...")
        for field_name, field_type, field_precision, field_alias in fields:
            if field_type == "DOUBLE":
                # 对于DOUBLE类型，设置精度和小数位数
                arcpy.AddField_management(
                    in_table=os.path.join(workspace, table_name),
                    field_name=field_name,
                    field_type=field_type,
                    field_precision=15,  # 总精度
                    field_scale=3,       # 小数位数
                    field_alias=field_alias
                )
            else:
                # 对于其他类型，使用标准AddField
                arcpy.AddField_management(
                    in_table=os.path.join(workspace, table_name),
                    field_name=field_name,
                    field_type=field_type,
                    field_alias=field_alias
                )
            print(f"  已添加字段: {field_name} ({field_type})")
        
        print(f"\n表 {table_name} 创建成功！")
        print(f"完整路径: {os.path.join(workspace, table_name)}")
        
    except arcpy.ExecuteError:
        print(f"ArcPy错误: {arcpy.GetMessages(2)}")
        return None
    except Exception as e:
        print(f"错误: {str(e)}")
        return None
def calculate_mean_timeseries(timeseries_dict):
    """
    计算多个时间序列的均值序列
    
    参数:
    timeseries_dict (dict): 字典，其中键是时间序列的编号，值是包含时序数据的列表
    
    返回:
    list: 包含所有时间序列均值的列表
    dict: 包含各种统计信息的字典
    """
    arcpy.AddMessage(f"输入字典包含 {len(timeseries_dict)} 个时间序列")
    
    # 检查所有时间序列的长度
    lengths = [len(series) for series in timeseries_dict.values()]
    
    if not lengths:
        arcpy.AddMessage("错误: 输入字典为空")
        return [], {"error": "输入字典为空"}
    
    # 检查所有时间序列是否有相同的长度
    if len(set(lengths)) > 1:
        arcpy.AddMessage(f"警告: 时间序列长度不一致，范围从 {min(lengths)} 到 {max(lengths)}")
        # 决定使用的长度(这里使用最小长度，也可以选择其他策略)
        use_length = min(lengths)
        arcpy.AddMessage(f"将使用最小长度: {use_length}")
    else:
        use_length = lengths[0]
        arcpy.AddMessage(f"所有时间序列长度一致: {use_length}")
    
    # 创建一个二维数组来存储所有时间序列数据
    # 只使用每个序列的前use_length个值
    data_array = np.array([series[:use_length] for series in timeseries_dict.values()])
    
    # 沿着第一个轴(序列轴)计算均值，得到每个时间点的均值
    mean_timeseries = np.mean(data_array, axis=0).tolist()
    rounded_timeseries_list = [round(value, 3) for value in mean_timeseries]
    
    # 计算额外的统计信息
    stats = {
        "count": len(timeseries_dict),
        "original_length_min": min(lengths),
        "original_length_max": max(lengths),
        "used_length": use_length,
        "mean_min": min(rounded_timeseries_list),
        "mean_max": max(rounded_timeseries_list),
        "std_timeseries": np.std(data_array, axis=0).tolist(),  # 每个时间点的标准差
        "median_timeseries": np.median(data_array, axis=0).tolist(),  # 每个时间点的中位数
    }
    
    arcpy.AddMessage(f"均值时间序列计算完成，长度为 {len(rounded_timeseries_list)}")
    
    return rounded_timeseries_list, stats
def script_tool(time_series_tb, location_field, datetime_field, sequence_field, value_field, regionalization_layer, region_id_field, subregion_code_field, curve_type, output_curve_tb):
    """Script code goes below"""
    # 设置工作空间环境
    default_ws= arcpy.env.workspace
    # 获取字段列表
    fields = [location_field, datetime_field, sequence_field, value_field]
    # 使用pandas DataFrame来处理数据
    # 首先读取数据到DataFrame
    data = []
    with arcpy.da.SearchCursor(time_series_tb, fields) as cursor:
        for row in cursor:
            data.append(row)
    df_timeseries = pd.DataFrame(data, columns=fields)
    # 确保datetime_field是datetime类型
    df_timeseries[datetime_field] = pd.to_datetime(df_timeseries[datetime_field])
    # 确保其他字段类型正确
    df_timeseries[location_field] = df_timeseries[location_field].astype(int)
    df_timeseries[sequence_field] = df_timeseries[sequence_field].astype(int)
    df_timeseries[value_field] = df_timeseries[value_field].astype(float)
    # 获取字段列表
    fields = [region_id_field, subregion_code_field]
    # 创建空列表来存储数据
    data = []
    # 使用SearchCursor读取表格数据
    with arcpy.da.SearchCursor(regionalization_layer, fields) as cursor:
        for row in cursor:
            data.append(row)
    # 创建DataFrame
    df_regions = pd.DataFrame(data, columns=fields)
    # 确保字段类型正确（long类型对应Python中的int）
    df_regions[region_id_field] = df_regions[region_id_field].astype(int)
    df_regions[subregion_code_field] = df_regions[subregion_code_field].astype(int)
    df_merged = pd.merge(
        left=df_timeseries,
        right=df_regions,
        left_on=location_field,
        right_on=region_id_field,
        how='left'
    )
    arcpy.AddMessage(df_merged.head(40))
    nested_dict = {}
    # 遍历所有唯一的子区域ID
    for subregion in df_merged[subregion_code_field].unique():
        # 为每个子区域创建一个字典
        nested_dict[subregion] = {}
        
        # 获取该子区域的所有数据
        subregion_data = df_merged[df_merged[subregion_code_field] == subregion]
        
        # 遍历该子区域的所有唯一区域ID
        for region in subregion_data[region_id_field].unique():
            # 获取该区域的所有数据
            region_data = subregion_data[subregion_data[region_id_field] == region]
            
            # 按sequence_field排序
            sorted_data = region_data.sort_values(by=sequence_field)
            
            # 提取排序后的均值列表
            mean_values = sorted_data[value_field].tolist()
            
            # 存储到嵌套字典中
            nested_dict[subregion][region] = mean_values
    # 验证结果
    arcpy.AddMessage("二级嵌套字典创建完成!")
    arcpy.AddMessage(f"总共包含 {len(nested_dict)} 个子区域")
    # 打印字典结构的总体统计
    region_counts = [len(regions) for regions in nested_dict.values()]
    total_regions = sum(region_counts)
    avg_regions_per_subregion = total_regions / len(nested_dict) if nested_dict else 0
    arcpy.AddMessage(f"\n字典统计:")
    arcpy.AddMessage(f"- 总子区域数: {len(nested_dict)}")
    arcpy.AddMessage(f"- 总区域数: {total_regions}")
    arcpy.AddMessage(f"- 平均每个子区域包含 {avg_regions_per_subregion:.2f} 个区域")
    # 方法4: 完整遍历并访问列表内容
    arcpy.AddMessage("方法4: 完整遍历并显示列表内容(仅前3个值)")
    for subregion, regions in nested_dict.items():
        arcpy.AddMessage(f"子区域 {subregion}:")
        for region, values in regions.items():
            # 只显示前3个值，避免输出过多
            preview = values[:3] if len(values) > 3 else values
            arcpy.AddMessage(f"  区域 {region}: {preview}...")
    create_table_with_fields(output_curve_tb)
    ts_list = extract_sorted_dt_list(df_merged, subregion_code_field, region_id_field, datetime_field)
    datetime_list_converted = [ts.strftime('%Y/%m/%d %H:%M:%S') for ts in ts_list]
    arcpy.AddMessage(datetime_list_converted)
    if curve_type == "Mean_Time_Series":
        for subregion, regions_dict in nested_dict.items():
            dt_seq = 0
            r = calculate_mean_timeseries(regions_dict)
            for val_item in r[0]:
                temp_ts = datetime.datetime.strptime(datetime_list_converted[dt_seq], '%Y/%m/%d %H:%M:%S')
                arcpy.AddMessage(temp_ts)
                add_single_row(output_curve_tb, subregion, temp_ts ,val_item)
                dt_seq = dt_seq + 1
            arcpy.AddMessage(r[0])
            arcpy.AddMessage(r[1])
    if curve_type == "Representive_Time_Series":
        for subregion1, regions_dict1 in nested_dict.items():
            dt_seq = 0
            arcpy.AddMessage(regions_dict1)
            representative_id, representative_ts, score = find_most_representative_timeseries(regions_dict1, distance_metric='euclidean', normalize=True, plot_result=True)
            arcpy.AddMessage("++++++")
            arcpy.AddMessage(representative_id)
            arcpy.AddMessage(representative_ts)
            for val_item in representative_ts:
                temp_ts = datetime.datetime.strptime(datetime_list_converted[dt_seq], '%Y/%m/%d %H:%M:%S')
                arcpy.AddMessage(temp_ts)
                add_single_row(output_curve_tb, representative_id, temp_ts ,val_item)
                dt_seq = dt_seq + 1
        
        
    
    
    return
if __name__ == "__main__":
    time_series_tb = arcpy.GetParameterAsText(0)
    location_field = arcpy.GetParameterAsText(1)
    datetime_field = arcpy.GetParameterAsText(2)
    sequence_field = arcpy.GetParameterAsText(3)
    value_field = arcpy.GetParameterAsText(4)
    regionalization_layer = arcpy.GetParameterAsText(5)
    region_id_field = arcpy.GetParameterAsText(6)
    subregion_code_field = arcpy.GetParameterAsText(7)
    curve_type = arcpy.GetParameterAsText(8)
    output_curve_tb = arcpy.GetParameterAsText(9)
    script_tool(time_series_tb, location_field, datetime_field, sequence_field, value_field, regionalization_layer, region_id_field, subregion_code_field, curve_type, output_curve_tb)
