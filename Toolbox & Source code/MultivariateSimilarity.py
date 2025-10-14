import numpy as np
from scipy import signal
from datetime import datetime, timedelta
import math
import bisect
from typing import List, Tuple


# 计算欧式距离
def compute_Edis_distance(ts1, ts2):
    # 提取值序列
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]

    # 确定较短序列的长度
    min_length = min(len(values1), len(values2))

    # 计算匹配部分的相似性
    matching_similarity = sum(abs(values1[i] - values2[i]) for i in range(min_length))

    # 计算平均相似性
    average_similarity = matching_similarity / min_length

    # 计算不匹配部分的长度
    unmatched_length = abs(len(values1) - len(values2))

    # 计算不匹配部分的相似性
    unmatched_similarity = average_similarity * unmatched_length

    # 计算总相似性
    total_similarity = matching_similarity + unmatched_similarity

    return total_similarity

# DTW度量两个序列的差异性，从时间序列中只提取值，忽略时间戳
def compute_dtw_distance(ts1, ts2):
    seq1 = [item[0] for item in ts1]
    seq2 = [item[0] for item in ts2]

    n, m = len(seq1), len(seq2)

    # 初始化成本矩阵
    # 第一行和第一列设置为无穷大，除了(0,0)位置
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    # 填充成本矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            try:
            # 计算当前两个点之间的距离
                cost = abs(seq1[i - 1] - seq2[j - 1])
            except Exception  as e:
                print(e)
            # 选择最小成本路径
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j],  # 插入
                                           cost_matrix[i, j - 1],  # 删除
                                           cost_matrix[i - 1, j - 1])  # 匹配

    # 返回右下角的值，即总的DTW距离
    return cost_matrix[n, m]

# 计算两个不规则采样时间序列的周期差异性
def compute_period_distance(ts1, ts2, fmin=0.1, fmax=10, n_freqs=1000):
    """
    计算两个不规则采样时间序列的周期相似性

    参数:
    ts1, ts2: 两个输入的时间序列
    fmin, fmax: 频率范围的最小值和最大值
    n_freqs: 要计算的频率点数

    返回:
    similarity: 周期相似性得分
    frequencies: 频率数组
    power1_norm, power2_norm: 归一化的功率谱密度
    timestamps1, timestamps2: 原始时间戳
    values1, values2: 原始数值
    """

    def preprocess_timeseries(ts):
        """
        预处理时间序列数据

        参数:
        ts: 输入的时间序列，格式为[(value, timestamp), ...]

        返回:
        values: 数值数组
        timestamps: 对应的datetime对象数组
        """
        values, timestamps = zip(*ts)
        values = np.array(values)
        timestamps = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps])
        return values, timestamps

    # 预处理时间序列
    values1, timestamps1 = preprocess_timeseries(ts1)
    values2, timestamps2 = preprocess_timeseries(ts2)

    # 将时间戳转换为相对秒数
    timestamps1_seconds = np.array([(t - timestamps1[0]).total_seconds() for t in timestamps1])
    timestamps2_seconds = np.array([(t - timestamps2[0]).total_seconds() for t in timestamps2])

    # 计算时间序列的特性
    duration1 = timestamps1_seconds[-1] - timestamps1_seconds[0]
    duration2 = timestamps2_seconds[-1] - timestamps2_seconds[0]

    min_interval1 = np.min(np.diff(timestamps1_seconds))
    min_interval2 = np.min(np.diff(timestamps2_seconds))

    # 根据时间序列特性调整频率范围
    fmin = max(fmin, 1 / max(duration1, duration2))
    fmax = min(fmax, 1 / (2 * min(min_interval1, min_interval2)))

    # 创建频率数组
    frequencies = np.linspace(fmin, fmax, n_freqs)

    # 计算Lomb-Scargle周期图
    power1 = signal.lombscargle(timestamps1_seconds, values1, frequencies)
    power2 = signal.lombscargle(timestamps2_seconds, values2, frequencies)

    # 归一化功率谱
    power1_norm = power1 / np.sum(power1)
    power2_norm = power2 / np.sum(power2)

    # 计算余弦相似度作为周期相似性得分
    similarity = np.dot(power1_norm, power2_norm) / (np.linalg.norm(power1_norm) * np.linalg.norm(power2_norm))

    # 将相似性转换为不相似性
    dissimilarity = 1 - similarity

    # return dissimilarity, frequencies, power1_norm, power2_norm, timestamps1, timestamps2, values1, values2  # 用于测试
    return dissimilarity

# 度量形状差异性
def calculate_shape_distance(TS_1, TS_2):
    def parse_datetime(date_string):
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # 对两个序列进行等序列化
    def equalize_time_series(ts1, ts2):
        ts1 = [(v, parse_datetime(t)) for v, t in ts1]
        ts2 = [(v, parse_datetime(t)) for v, t in ts2]

        def find_time_intersection(ts1, ts2):
            start1, end1 = ts1[0][1], ts1[-1][1]
            start2, end2 = ts2[0][1], ts2[-1][1]
            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)
            return intersection_start, intersection_end

        def interpolate_value(t, t1, t2, v1, v2):
            if t1 == t2:
                return v1
            ratio = ((t - t1).total_seconds() / 60) / ((t2 - t1).total_seconds() / 60)
            return v1 + ratio * (v2 - v1)

        intersection_start, intersection_end = find_time_intersection(ts1, ts2)
        all_timestamps = sorted(set([t for _, t in ts1 + ts2 if intersection_start <= t <= intersection_end]))

        def interpolate_series(ts):
            result = []
            for t in all_timestamps:
                if t < ts[0][1] or t > ts[-1][1]:
                    continue
                i = bisect.bisect_left([x[1] for x in ts], t)
                if i == 0 or ts[i][1] == t:
                    result.append((ts[i][0], t))
                else:
                    v = interpolate_value(t, ts[i - 1][1], ts[i][1], ts[i - 1][0], ts[i][0])
                    result.append((v, t))
            return result

        ts1_eq = interpolate_series(ts1)
        ts2_eq = interpolate_series(ts2)

        min_len = min(len(ts1_eq), len(ts2_eq))
        ts1_eq = ts1_eq[:min_len]
        ts2_eq = ts2_eq[:min_len]

        ts1_eq = [(v, t.strftime("%Y-%m-%d %H:%M:%S")) for v, t in ts1_eq]
        ts2_eq = [(v, t.strftime("%Y-%m-%d %H:%M:%S")) for v, t in ts2_eq]

        return ts1_eq, ts2_eq

    def calculate_similarity(ts1, ts2):
        patterns1, time_lengths1, _ = pattern_analysis(ts1)
        patterns2, time_lengths2, _ = pattern_analysis(ts2)

        min_length = min(len(patterns1), len(patterns2))
        patterns1 = patterns1[:min_length]
        patterns2 = patterns2[:min_length]
        time_lengths1 = time_lengths1[:min_length]
        time_lengths2 = time_lengths2[:min_length]

        total_time = sum(time_lengths1)
        dis_similarity = 0

        for i in range(min_length):
            t_wi = time_lengths1[i] / total_time
            pattern_diff = abs(patterns1[i] - patterns2[i])
            value_diff = abs((ts1[i + 2][0] - ts1[i + 1][0]) - (ts2[i + 2][0] - ts2[i + 1][0]))
            # dis_similarity += t_wi * pattern_diff * value_diff
            dis_similarity += t_wi * pattern_diff# * value_diff
        return dis_similarity

    def pattern_analysis(time_series, epsilon=math.tan(math.pi / 360)):
        def calculate_slope(t1, t2, v1, v2):
            time_diff = (parse_datetime(t2) - parse_datetime(t1)).total_seconds() / 60
            return (v2 - v1) / time_diff if time_diff != 0 else 0

        def determine_pattern(k_current, k_next, epsilon):
            if (k_next > epsilon and k_current < epsilon) or (
                    k_next > epsilon and k_current > epsilon and k_next - k_current > 0):
                return 3  # 加速上升
            elif k_next > epsilon and k_current > epsilon and k_next - k_current == 0:
                return 2  # 水平上升
            elif k_next > epsilon and k_current > epsilon and k_next - k_current < 0:
                return 1  # 减速上升
            elif (k_next < -epsilon and k_current > -epsilon) or (
                    k_next < -epsilon and k_current < -epsilon and k_next - k_current < 0):
                return -3  # 加速下降
            elif k_next < -epsilon and k_current < -epsilon and k_next - k_current == 0:
                return -2  # 水平下降
            elif k_next < -epsilon and k_current < -epsilon and k_next - k_current > 0:
                return -1  # 减速下降
            elif -epsilon < k_next < epsilon:
                return 0  # 不变
            else:
                return None  # 未定义模式

        patterns = []
        time_lengths = []
        slopes = []

        for i in range(len(time_series) - 2):
            v1, t1 = time_series[i]
            v2, t2 = time_series[i + 1]
            v3, t3 = time_series[i + 2]

            k_current = calculate_slope(t1, t2, v1, v2)
            k_next = calculate_slope(t2, t3, v2, v3)

            pattern = determine_pattern(k_current, k_next, epsilon)
            if pattern is not None:
                patterns.append(pattern)
                time_lengths.append((parse_datetime(t2) - parse_datetime(t1)).total_seconds() / 60)
                slopes.append(k_current)

        return patterns, time_lengths, slopes

    # 等序列化
    TS_1_eq, TS_2_eq = equalize_time_series(TS_1, TS_2)

    # 模式分析
    patterns1, _, slopes1 = pattern_analysis(TS_1_eq)
    patterns2, _, slopes2 = pattern_analysis(TS_2_eq)

    # 计算相似度
    dis_similarity = calculate_similarity(TS_1_eq, TS_2_eq)
    # print(f"两个序列的形状相似度: {similarity}")
    return dis_similarity

# 从给定的文件路径读取时序信息，创建并返回时序列表。
def create_lengthwise_time_series_list_by_specific_column(file_path, id_col=0, value_col=1, date_col=True, scale_rate=1):
    """
    数据格式：area_id,value1,value2,value3,...,time_series

    参数:
    file_path (str): 包含时序信息的文件路径
    id_col (int): 指定id所在的列
    value_col (int): 指定值的数量
    date_col (int): 是否包含时间列
    scale_rate (int): 指定缩放比例

    返回:
    list: 包含所有时序序列的列表，每个序列是一个列表的元组 (值, 时间字符串)
    """

    # 根据给定的压缩比压缩时间序列。
    def compress_time_series(ts: List[Tuple[int, str]], scale_rate: float) -> List[Tuple[int, str]]:
        """
        根据给定的压缩比压缩时间序列。

        :param ts: 原始时间序列，格式为 (值, 时间戳) 的元组列表
        :param scale_rate: 压缩比 (0 < scale_rate <= 1)
        :return: 压缩后的时间序列
        """

        # 基于周围的时间戳插值计算新的时间戳。
        def interpolate_value(prev: float, current: float, next: float) -> float:
            """
            基于周围的值插值计算新的值。

            :param prev: 前一个值
            :param current: 当前值
            :param next: 后一个值
            :return: 插值后的新值
            """
            return round(float((prev + current + next) / 3), 3)

        # 基于周围的时间戳插值计算新的时间戳。
        def interpolate_timestamp(prev: str, current: str, next: str) -> str:
            """
            基于周围的时间戳插值计算新的时间戳。

            :param prev: 前一个时间戳
            :param current: 当前时间戳
            :param next: 后一个时间戳
            :return: 插值后的新时间戳
            """
            # 将字符串时间戳转换为 datetime 对象
            prev_dt = datetime.strptime(prev, "%Y-%m-%d %H:%M:%S")
            current_dt = datetime.strptime(current, "%Y-%m-%d %H:%M:%S")
            next_dt = datetime.strptime(next, "%Y-%m-%d %H:%M:%S")

            # 计算插值后的时间戳
            interpolated_dt = prev_dt + (next_dt - prev_dt) / 2
            return interpolated_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 检查压缩比是否有效
        if not 0 < scale_rate <= 1:
            raise ValueError("压缩比 scale_rate 必须在 0 和 1 之间")

        # 如果时间序列长度小于等于2，无需压缩，直接返回
        if len(ts) <= 2:
            return ts

        # 计算需要保留的元素数量
        n = len(ts)
        keep_count = max(2, int(n * scale_rate))

        # 始终保留第一个元素
        result = [ts[0]]

        if keep_count > 2:
            # 计算选择元素的步长
            step = (n - 2) / (keep_count - 2)

            # 对中间的元素进行插值
            for i in range(1, keep_count - 1):
                index = int(i * step)
                prev_index = int((i - 1) * step)
                next_index = min(int((i + 1) * step), n - 1)
                # 插值计算新的值
                value = interpolate_value(ts[prev_index][0], ts[index][0], ts[next_index][0])

                # 插值计算新的时间戳
                timestamp = interpolate_timestamp(ts[prev_index][1], ts[index][1], ts[next_index][1])

                result.append((value, timestamp))

        # 添加最后一个元素
        result.append(ts[-1])
        return result

    # 初始化一个字典来存储所有的时序序列
    series_dict = {}

    # 设置起始时间（用于两列数据的情况）
    start_time = datetime(2024, 1, 1, 0, 0)

    try:
        # 尝试打开并读取文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                if not line.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    continue
                # 去除行首尾的空白字符，并按逗号分割
                items = line.strip().split(',')

                # 如果分割后的列表少于2个或多于3个元素，则跳过此行
                if len(items) < 2 :
                    print(f"警告: 时序文件跳过格式不正确的行: {line.strip()}")
                    continue

                # 提取序列编号和值
                sequence_number = int(items[id_col])


                # 处理时间信息
                if date_col == True:
                    # 如果数据本身包含时间序列，直接使用提供的时间
                    time_str = items[-1]
                    # 获取有时间的所有值的列
                    value_lst = [float(v) for v in items[1:-1]]# 将值转换为小数


                else:
                    # 如果数据本身不含时间序列，生成时间
                    if sequence_number not in series_dict: # 如果是新序列，使用起始时间
                        current_time = start_time
                    else:
                        # 如果序列已存在，在最后一个时间基础上加5分钟
                        last_time = datetime.strptime(series_dict[sequence_number][-1][1], "%Y-%m-%d %H:%M:%S")
                        current_time = last_time + timedelta(minutes=5)
                    # 获取没有时间的所有值的列
                    value_lst = [float(v) for v in items[1:]]  # 将值转换为小数

                    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                # 获取一行记录中值的数量
                value_len = len(value_lst)
                # 将 (值, 时间) 元组添加到对应的序列
                if sequence_number in series_dict:
                    for i,v in enumerate(value_lst):
                        series_dict[sequence_number][i].append((round(v,3), time_str))

                else:
                    series_dict[sequence_number] = [[(round(v,3), time_str)] for v in value_lst]

    except FileNotFoundError:
        # 如果文件不存在，打印错误信息
        print(f"错误：未找到文件 '{file_path}'。")
        return []
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息m
        print(f"错误：无法读取文件 '{file_path}'。")
        return []
    except ValueError as e:     #ValueError
        # 如果在转换过程中出现错误（例如，非整数值），打印错误信息
        print(f"错误：处理数据时出现问题 - {str(e)}")
        return []

    # 根据给定的压缩比压缩时间序列。
    scaled_series_dict = {}
    for index, time_series in series_dict.items():
        scaled_series_dict[index] = []
        for i in range(value_len):
            scaled_time_series = compress_time_series(time_series[i], scale_rate)
            scaled_series_dict[index].append(scaled_time_series)
    return scaled_series_dict


# 创建权重矩阵
def create_weight_matrix(scaled_series_dict,weight_type='Edis'):
    weight_type_dict = {'edis':"compute_edis_distance", 'period':'compute_period_distance', 'dtw':'compute_dtw_distance', 'shape':'calculate_shape_distance'}
    key_list = scaled_series_dict.keys()
    key_max_value = max(key_list)+1
    dis_matrix = np.zeros((key_max_value, key_max_value)) -1
    # 获取一个区域中研究对象的数量
    value_len = len(scaled_series_dict[list(key_list)[0]])
    for i in key_list:
        for j in key_list:
            if i != j and dis_matrix[i][j] == -1:
                sum_dis = 0
                for k in range(value_len):
                    sum_dis += eval(weight_type_dict[weight_type])(scaled_series_dict[i][k], scaled_series_dict[j][k])
                dis_matrix[i][j] = dis_matrix[j][i] = sum_dis
    max_dis = np.max(dis_matrix)
    dis_matrix[dis_matrix == -1] = max_dis
    sim_matrix = max_dis - dis_matrix
    return sim_matrix


# 将对称权重矩阵写入文件
def write_weight_to_file(weight_matrix, file):
    """
    参数:
        weight_matrix: numpy.ndarray - 对称权重矩阵
        file: str - 输出文件路径
    """

    # 检查矩阵是否对称
    if not np.allclose(weight_matrix, weight_matrix.T):
        raise ValueError("权重矩阵必须是对称的")

    with open(file, 'w') as f:
        # 写入标题行
        f.write("n1,n2,weight\n")

        # 遍历上三角矩阵 (i < j)
        for i in range(weight_matrix.shape[0]):
            for j in range(i + 1, weight_matrix.shape[1]):
                weight = round(weight_matrix[i, j],4)   # 保留4位小数
                if weight != 0:  # 只写入非零权重
                    f.write(f"{i},{j},{weight}\n")





if __name__ == "__main__":
    # 小数据使用示例
    TS_1 = [(3, "2024-04-01 05:20"), (6, "2024-04-01 05:40"), (16, "2024-04-01 06:20"), (8, "2024-04-01 06:35"),
            (3, "2024-04-01 07:10"), (6, "2024-04-01 7:40"), (16, "2024-04-01 8:20"), (8, "2024-04-01 8:30")]
    TS_2 = [(3, "2024-04-01 05:35"), (15, "2024-04-01 05:50"), (6, "2024-04-01 06:20"), (18, "2024-04-01 07:00"),
            (3, "2024-04-01 07:20"), (15, "2024-04-01 7:50"), (6, "2024-04-01 8:10"), (18, "2024-04-01 8:40")]
    # print(compute_period_distance(TS_1, TS_2))

    # 真实数据文件
    file_path = r'data_test/多变量数据集.csv'
    #读取数据文件
    scaled_series_dict = create_lengthwise_time_series_list_by_specific_column(file_path, id_col=0, value_col=1, date_col=True, scale_rate=1)
    # 根据数据文件创建权重矩阵，权重越大表示越相似
    # 'edis', 'period', 'dtw', 'shape'
    weight_matrix = create_weight_matrix(scaled_series_dict,'dtw')
    write_weight_to_file(weight_matrix, "weights.csv")
    print(weight_matrix)

    # print(compute_Edis_distance(scaled_series_dict[1],scaled_series_dict[2]))
    # print(coumput_dtw_distance(scaled_series_dict[1],scaled_series_dict[2]))
    # print(calculate_shape_distance(scaled_series_dict[1],scaled_series_dict[2]))
    # print(compute_period_distance(scaled_series_dict[1],scaled_series_dict[2]))