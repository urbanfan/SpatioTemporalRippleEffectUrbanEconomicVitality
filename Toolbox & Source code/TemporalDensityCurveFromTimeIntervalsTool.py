import arcpy
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def create_table_from_dataframe(df, output_location, table_name, dt_field_type):
    """
    将Pandas DataFrame转换为ArcGIS表格
    
    参数:
    df: pandas DataFrame，包含group_id(int), dt(datetime), density_val(double)
    output_location: 输出位置(文件地理数据库或文件夹路径)
    table_name: 输出表名称
    
    返回:
    创建的表格的完整路径
    """
    arcpy.env.overwriteOutput = True

    arcpy.AddMessage(df.head(10))
    
    # 确保输出位置存在
    if not os.path.exists(output_location):
        arcpy.AddError(f"输出位置 {output_location} 不存在")
        return None
        
    # 构建完整的输出路径
    if output_location.endswith(".gdb"):
        out_table = os.path.join(output_location, table_name)
    else:
        out_table = os.path.join(output_location, f"{table_name}.dbf")
    
    # 创建表格
    arcpy.AddMessage("创建表格...")
    arcpy.management.CreateTable(output_location, table_name)
    
    # 添加字段
    arcpy.AddMessage("添加字段...")
    arcpy.management.AddField(out_table, "group_id", "LONG")
    arcpy.management.AddField(out_table, "seq", "LONG")
    if dt_field_type == "datetime":
        arcpy.management.AddField(out_table, "dt", "DATE")
    if dt_field_type == "date":
        arcpy.management.AddField(out_table, "dt", "DATEONLY")
    if dt_field_type == "time":
        arcpy.management.AddField(out_table, "dt", "TIMEONLY")
    arcpy.management.AddField(out_table, "density_val", "DOUBLE")
    
    # 准备插入数据
    arcpy.AddMessage(f"开始插入 {len(df)} 行数据...")
    
    # 使用插入游标
    with arcpy.da.InsertCursor(out_table, ["group_id", "seq", "dt", "density_val"]) as cursor:
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():

            try:
                dt = None
                # 处理可能的NaN值
                group_id = int(row['group_id']) if not pd.isna(row['group_id']) else None
                seq = int(row['seq']) if not pd.isna(row['seq']) else None
                if dt_field_type == "datetime":
                    dt = row['dt'].to_pydatetime() if not pd.isna(row['dt']) else None
                if dt_field_type == "time":
                    dt =  datetime.strptime(row['dt'], "%H:%M:%S").time() if not pd.isna(row['dt']) else None
                if dt_field_type == "date":
                    dt =  datetime.strptime(row['dt'], "%Y-%m-%d").time() if not pd.isna(row['dt']) else None
                density_val = float(row['density_val']) if not pd.isna(row['density_val']) else None
                
                # 插入一行数据
                cursor.insertRow([group_id, seq, dt, density_val])
                
                # 每1000行显示进度
                if index > 0 and index % 1000 == 0:
                    arcpy.AddMessage(f"已插入 {index} 行...")
                    
            except Exception as e:
                arcpy.AddWarning(f"插入第 {index} 行时出错: {str(e)}")
                arcpy.AddWarning(f"行数据: {row.to_dict()}")
                continue
    
    arcpy.AddMessage(f"成功创建表格 {out_table} 并插入 {len(df)} 行数据")
    return out_table

def interval_density(
    intervals,
    x_start,
    x_end,
    x_freq='1H'  # 步长：'1H'小时，'10min'分钟，'1D'天，'1M'月等
):
    """
    intervals: list of [start, end] 时间段，字符串或datetime格式
    x_start, x_end: 横坐标起止，字符串或datetime格式
    x_freq: 横坐标间隔，pandas支持的频率字符串
    """
    # 转换为datetime
    intervals = [
        [pd.to_datetime(start), pd.to_datetime(end)]
        for start, end in intervals
    ]
    x_start = pd.to_datetime(x_start)
    x_end = pd.to_datetime(x_end)

    # 生成横坐标分段
    bins = pd.date_range(start=x_start, end=x_end, freq=x_freq)
    if bins[-1] < x_end:
        bins = bins.append(pd.DatetimeIndex([x_end]))
    res = []
    labels = []
    for i in range(len(bins)-1):
        seg_start, seg_end = bins[i], bins[i+1]
        labels.append(seg_start)
        total_covered = 0   # 时长被覆盖的总秒数
        for (intv_start, intv_end) in intervals:
            # 计算时段与分段的交集
            overlap_start = max(seg_start, intv_start)
            overlap_end = min(seg_end, intv_end)
            if overlap_start < overlap_end:
                total_covered += (overlap_end - overlap_start).total_seconds()
        res.append(total_covered)
    return labels, res

def interval_density1(
    intervals,
    x_start,
    x_end,
    x_freq='1H'
):
    """
    intervals: list of [start, end]，可为 仅日期/仅时间/日期时间
    x_start, x_end: 支持同上
    x_freq: 分段间隔
    """

    #arcpy.AddMessage(intervals)
    #arcpy.AddMessage(x_start)
    #arcpy.AddMessage(x_end)
    #arcpy.AddMessage(x_freq)

    # 辅助函数：判断类型
    def detect_type(s):
        """
        判断输入字符串/时间对象的类型（'date', 'datetime', 'time'）
        支持 '0:10:22' 这种非标准时间写法。
        """
        # 如果是 pandas.Timestamp
        if isinstance(s, pd.Timestamp):
            if (s.hour, s.minute, s.second, s.microsecond) == (0, 0, 0, 0):
                return 'date'
            return 'datetime'
        
        s_str = str(s).strip()

        # 1. 先判断时间（如 0:10:22、12:34:56、23:59:59）
        time_pattern = re.compile(r'^([0-1]?\d|2[0-3]|\d):[0-5]?\d:[0-5]?\d$')
        if time_pattern.match(s_str):
            return 'time'

        # 2. 判断日期（如 2024-07-04）
        date_pattern = re.compile(r'^\d{4}-\d{1,2}-\d{1,2}$')
        if date_pattern.match(s_str):
            return 'date'

        # 3. 判断日期时间（如 2024-07-04 12:34:56）
        dt_pattern = re.compile(r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$')
        if dt_pattern.match(s_str):
            return 'datetime'

        # 4. 兜底尝试解析
        try:
            t = pd.to_datetime(s_str)
            if (t.hour, t.minute, t.second, t.microsecond) == (0, 0, 0, 0):
                return 'date'
            return 'datetime'
        except Exception:
            try:
                t = pd.to_datetime(s_str, format='%H:%M:%S')
                return 'time'
            except Exception:
                raise ValueError(f"无法识别类型: {s}")
    
    # 统一检测所有类型
    all_types = []
    
    i= 1
    for pair in intervals:
        all_types.append(detect_type(pair[0]))
       
        all_types.append(detect_type(pair[1]))
        
        i = i + 1
    all_types.append(detect_type(x_start))
    
    all_types.append(detect_type(x_end))
    
    uniq_types = set(all_types)
   
    if len(uniq_types) > 1:
        raise ValueError(f"所有时间类型必须一致，检测到类型: {uniq_types}")
    ttype = uniq_types.pop()  # 'date', 'datetime', 'time'

    # 仅时间的特殊处理
    if ttype == 'time':
        # 全部转成秒数
        def to_seconds(t):
            if isinstance(t, pd.Timestamp):
                return t.hour*3600 + t.minute*60 + t.second
            t = str(t)
            if len(t.split(':')) == 3:
                h, m, s = map(int, t.split(':'))
                return h*3600 + m*60 + s
            else:
                raise ValueError(f"无法解析时间: {t}")
        intervals_sec = [
            [to_seconds(start), to_seconds(end)] for start, end in intervals
        ]
        x_start_sec = to_seconds(x_start)
        x_end_sec = to_seconds(x_end)
        delta = pd.to_timedelta(x_freq).total_seconds()
        bins = np.arange(x_start_sec, x_end_sec+delta, delta)
        if bins[-1] > x_end_sec:
            bins[-1] = x_end_sec
        res = []
        labels = []
        for i in range(len(bins)-1):
            seg_start, seg_end = bins[i], bins[i+1]
            labels.append(f"{int(seg_start//3600):02d}:{int((seg_start%3600)//60):02d}:{int(seg_start%60):02d}")
            total_covered = 0
            for intv_start, intv_end in intervals_sec:
                overlap_start = max(seg_start, intv_start)
                overlap_end = min(seg_end, intv_end)
                if overlap_start < overlap_end:
                    total_covered += overlap_end - overlap_start
            res.append(total_covered)
        return labels, res, ttype

    else:
        # 日期或日期时间
        intervals = [
            [pd.to_datetime(start), pd.to_datetime(end)]
            for start, end in intervals
        ]
        x_start = pd.to_datetime(x_start)
        x_end = pd.to_datetime(x_end)
        bins = pd.date_range(start=x_start, end=x_end, freq=x_freq)
        if bins[-1] < x_end:
            bins = bins.append(pd.DatetimeIndex([x_end]))
        res = []
        labels = []
        for i in range(len(bins)-1):
            seg_start, seg_end = bins[i], bins[i+1]
            labels.append(seg_start)
            total_covered = 0
            for (intv_start, intv_end) in intervals:
                overlap_start = max(seg_start, intv_start)
                overlap_end = min(seg_end, intv_end)
                if overlap_start < overlap_end:
                    total_covered += (overlap_end - overlap_start).total_seconds()
            res.append(total_covered)
        return labels, res, ttype
    
def interval_count_density1(
    intervals,
    x_start,
    x_end,
    x_freq='1H'
):
    """
    intervals: list of [start, end]，可为 仅日期/仅时间/日期时间
    x_start, x_end: 支持同上
    x_freq: 分段间隔
    返回: labels（分段起点）, counts（每段内被覆盖的时段数）, ttype（类型）
    """

    # 辅助函数：判断类型
    def detect_type(s):
        if isinstance(s, pd.Timestamp):
            if (s.hour, s.minute, s.second, s.microsecond) == (0, 0, 0, 0):
                return 'date'
            return 'datetime'
        s_str = str(s).strip()
        time_pattern = re.compile(r'^([0-1]?\d|2[0-3]|\d):[0-5]?\d:[0-5]?\d$')
        if time_pattern.match(s_str):
            return 'time'
        date_pattern = re.compile(r'^\d{4}-\d{1,2}-\d{1,2}$')
        if date_pattern.match(s_str):
            return 'date'
        dt_pattern = re.compile(r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$')
        if dt_pattern.match(s_str):
            return 'datetime'
        try:
            t = pd.to_datetime(s_str)
            if (t.hour, t.minute, t.second, t.microsecond) == (0, 0, 0, 0):
                return 'date'
            return 'datetime'
        except Exception:
            try:
                t = pd.to_datetime(s_str, format='%H:%M:%S')
                return 'time'
            except Exception:
                raise ValueError(f"无法识别类型: {s}")

    # 统一检测所有类型
    all_types = []
    for pair in intervals:
        all_types.append(detect_type(pair[0]))
        all_types.append(detect_type(pair[1]))
    all_types.append(detect_type(x_start))
    all_types.append(detect_type(x_end))
    uniq_types = set(all_types)
    if len(uniq_types) > 1:
        raise ValueError(f"所有时间类型必须一致，检测到类型: {uniq_types}")
    ttype = uniq_types.pop()  # 'date', 'datetime', 'time'

    # 仅时间的特殊处理
    if ttype == 'time':
        def to_seconds(t):
            if isinstance(t, pd.Timestamp):
                return t.hour*3600 + t.minute*60 + t.second
            t = str(t)
            if len(t.split(':')) == 3:
                h, m, s = map(int, t.split(':'))
                return h*3600 + m*60 + s
            else:
                raise ValueError(f"无法解析时间: {t}")
        intervals_sec = [
            [to_seconds(start), to_seconds(end)] for start, end in intervals
        ]
        x_start_sec = to_seconds(x_start)
        x_end_sec = to_seconds(x_end)
        delta = pd.to_timedelta(x_freq).total_seconds()
        bins = np.arange(x_start_sec, x_end_sec+delta, delta)
        if bins[-1] > x_end_sec:
            bins[-1] = x_end_sec
        counts = []
        labels = []
        for i in range(len(bins)-1):
            seg_start, seg_end = bins[i], bins[i+1]
            labels.append(f"{int(seg_start//3600):02d}:{int((seg_start%3600)//60):02d}:{int(seg_start%60):02d}")
            count = 0
            for intv_start, intv_end in intervals_sec:
                overlap_start = max(seg_start, intv_start)
                overlap_end = min(seg_end, intv_end)
                if overlap_start < overlap_end:
                    count += 1
            counts.append(count)
        return labels, counts, ttype

    else:
        # 日期或日期时间
        intervals_dt = [
            [pd.to_datetime(start), pd.to_datetime(end)]
            for start, end in intervals
        ]
        x_start_dt = pd.to_datetime(x_start)
        x_end_dt = pd.to_datetime(x_end)
        bins = pd.date_range(start=x_start_dt, end=x_end_dt, freq=x_freq)
        if bins[-1] < x_end_dt:
            bins = bins.append(pd.DatetimeIndex([x_end_dt]))
        counts = []
        labels = []
        for i in range(len(bins)-1):
            seg_start, seg_end = bins[i], bins[i+1]
            labels.append(seg_start)
            count = 0
            for (intv_start, intv_end) in intervals_dt:
                overlap_start = max(seg_start, intv_start)
                overlap_end = min(seg_end, intv_end)
                if overlap_start < overlap_end:
                    count += 1
            counts.append(count)
        return labels, counts, ttype

def group_times_by_id(input_table, start_field, end_field, group_field):
    """
    Read a table and group time pairs by group_id.
    
    Parameters:
    -----------
    input_table : str
        Path to the input table
    start_field : str
        Field name for start time
    end_field : str
        Field name for end time
    group_field : str
        Field name for group id
    
    Returns:
    --------
    dict
        Dictionary with group_id as key and list of [start_time, end_time] pairs as value
    """
    # Check if the table exists
    if not arcpy.Exists(input_table):
        arcpy.AddError(f"Input table {input_table} does not exist.")
        return None
    
    # Check if the fields exist
    field_names = [f.name for f in arcpy.ListFields(input_table)]
    for field in [start_field, end_field, group_field]:
        if field not in field_names:
            arcpy.AddError(f"Field {field} does not exist in {input_table}.")
            return None
    
    # Initialize the result dictionary
    result_dict = {}
    
    # Read the table
    with arcpy.da.SearchCursor(input_table, [start_field, end_field, group_field]) as cursor:
        for row in cursor:
            start_time = row[0]
            end_time = row[1]
            group_id = row[2]
            
            # Initialize the list for this group_id if it doesn't exist
            if group_id not in result_dict:
                result_dict[group_id] = []
            
            # Add the time pair to the list
            result_dict[group_id].append([start_time, end_time])
    
    return result_dict

def script_tool(input_time_interval_tb, start_time_field, end_time_field, group_field, method_type, start_refered_time, end_refered_time, time_interval, output_temporal_density_tb):
    """Script code goes below"""

    time_unit_dic = { 'Milliseconds': 'ms', 'Seconds': 'S', 'Minutes': 'min', 'Hours': 'H', 'Dyas': 'D', 'Weeks': 'W', 'Months': 'M', 'Years': 'Y'}

    arcpy.AddMessage(f"{start_refered_time},{end_refered_time},{time_interval}")

    # Decades Centuaries
    
    interval_val = time_interval.split(' ')[0]
    interval_unit = time_interval.split(' ')[1]

    arcpy.AddMessage(f"interval value: {interval_val}, interval unit: {interval_unit}")

    if interval_unit == "Decades" or interval_unit == "Centuaries" or interval_unit == "Unknown":
        return
    else:
        interval_unit = time_unit_dic[interval_unit]
        
    new_time_interval = interval_val + interval_unit

    time_od_dic = group_times_by_id(input_time_interval_tb, start_time_field, end_time_field, group_field)

    

    # 创建一个空的DataFrame，指定列名和数据类型
    df = pd.DataFrame(columns=['group_id', 'seq', 'dt', 'density_val'])

    # 设置列的数据类型
    df['group_id'] = df['group_id'].astype('int64')
    df['seq'] = df['seq'].astype('int64')
    df['dt'] = pd.to_datetime(df['dt'])
    df['density_val'] = df['density_val'].astype('float64')

    dt_field_type = ''

    for key, val in time_od_dic.items():
        
        if method_type == "Total_Covered_Time":

            time_label, density_val, dt_field_type = interval_density1(val, start_refered_time, end_refered_time, new_time_interval)

        if method_type == "Number_Covered_Intervals":
            time_label, density_val, dt_field_type = interval_count_density1(val, start_refered_time, end_refered_time, new_time_interval)

        group_label = [key] * len(time_label)
        time_sequence = list(range(1, len(time_label) + 1))

        # 将zip对象转换为新的DataFrame
        new_rows = pd.DataFrame(zip(group_label, time_sequence, time_label, density_val), columns=['group_id', 'seq', 'dt', 'density_val'])

        # 使用concat将新行添加到现有DataFrame
        df = pd.concat([df, new_rows], ignore_index=True)
    
    table_path = os.path.dirname(output_temporal_density_tb)
    table_name = os.path.basename(output_temporal_density_tb)
    create_table_from_dataframe(df, table_path, table_name, dt_field_type)

    return



if __name__ == "__main__":

    input_time_interval_tb = arcpy.GetParameterAsText(0)
    start_time_field = arcpy.GetParameterAsText(1)
    end_time_field = arcpy.GetParameterAsText(2)
    group_field = arcpy.GetParameterAsText(3)
    method_type = arcpy.GetParameterAsText(4)
    start_refered_time = arcpy.GetParameterAsText(5)
    end_refered_time = arcpy.GetParameterAsText(6)
    time_interval = arcpy.GetParameterAsText(7)
    output_temporal_density_tb = arcpy.GetParameterAsText(8)

    arcpy.AddMessage(start_refered_time)
    arcpy.AddMessage(end_refered_time)

    script_tool(input_time_interval_tb, start_time_field, end_time_field, group_field, method_type, start_refered_time, end_refered_time, time_interval, output_temporal_density_tb)
