import cv2
from pathlib import Path
import numpy as np
import os

INPUT_PATH = "./input_images"
OUTPUT_PATH = "./output_images"
DEBUG_PATH = "./debug_images"  # 新增debug输出路径
H = 2600
W = 1800

# 创建debug目录
Path(DEBUG_PATH).mkdir(parents=True, exist_ok=True)

def enhance_saturation(img: np.ndarray, saturation_factor: float = 2.0) -> np.ndarray:
    """增强图像饱和度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    # 增强饱和度通道
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adaptive_red_detection(img: np.ndarray) -> np.ndarray:
    """
    增强版红色圆点检测（结合颜色、形状和大小特征）
    返回红色圆点坐标数组
    """
    # 方法1：HSV空间检测 - 使用严格的红色范围
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 自适应饱和度阈值计算
    sat_channel = hsv[:, :, 1]
    sat_thresh_value, _ = cv2.threshold(sat_channel, 0, 255, cv2.THRESH_OTSU)
    min_sat = max(40, int(sat_thresh_value * 0.5))  # 提高饱和度阈值
    
    # 自适应亮度阈值计算
    val_channel = hsv[:, :, 2]
    val_thresh_value, _ = cv2.threshold(val_channel, 0, 255, cv2.THRESH_OTSU)
    min_val = max(80, int(val_thresh_value * 0.6))  # 提高亮度阈值
    
    # 定义严格的红色范围
    minred1 = np.array([0, min_sat, min_val])       # 低色调红色范围
    maxred1 = np.array([8, 255, 255])               # 缩小色调上限
    
    minred2 = np.array([172, min_sat, min_val])     # 高色调红色范围
    maxred2 = np.array([180, 255, 255])             # 缩小色调下限
    
    # 创建HSV掩码
    mask1 = cv2.inRange(hsv, minred1, maxred1)
    mask2 = cv2.inRange(hsv, minred2, maxred2)
    mask_hsv = cv2.bitwise_or(mask1, mask2)
    
    # 方法2：RGB空间检测 - 严格的红色检测
    b, g, r = cv2.split(img.astype(np.int16))
    
    # 计算红色通道优势
    red_advantage = r - np.maximum(b, g)
    
    # 自适应红色优势阈值
    red_thresh_value, _ = cv2.threshold(red_advantage.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    min_red_advantage = max(40, int(red_thresh_value * 0.7))  # 提高红色优势阈值
    
    # 创建RGB掩码
    red_mask = (red_advantage > min_red_advantage).astype(np.uint8) * 255
    
    # 合并两种检测结果 - 使用与操作提高精度
    combined_mask = cv2.bitwise_and(mask_hsv, red_mask)
    
    # 形态学优化 - 使用圆形核保持圆形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
    
    # 连通域分析去除小面积噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
    
    # 创建过滤后的掩码
    filtered_mask = np.zeros_like(processed_mask)
    valid_regions = []
    
    for i in range(1, num_labels):  # 跳过背景(0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 初步过滤：面积、长宽比
        if area > 100 and 0.7 < (width / height) < 1.3:
            filtered_mask[labels == i] = 255
            valid_regions.append(i)
    
    # 中值滤波去除椒盐噪点
    filtered_mask = cv2.medianBlur(filtered_mask, 5)
    
    # 二次形态学优化
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)
    
    # 使用过滤后的掩码进行轮廓检测
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 跳过太小的区域
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        # 1. 圆形度计算
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 2. 最小外接圆计算
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)
        
        # 3. 轮廓面积与外接圆面积比
        area_ratio = area / circle_area if circle_area > 0 else 0
        
        # 4. 紧凑度计算（基于最小外接矩形）
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        min_side = min(w, h)
        max_side = max(w, h)
        aspect_ratio = min_side / max_side if max_side > 0 else 0
        
        # 5. 凸性缺陷分析（检测不规则形状）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 复合圆形检测条件
        is_circle = (
            circularity > 0.7 and 
            area_ratio > 0.7 and 
            aspect_ratio > 0.8 and 
            solidity > 0.9
        )
        
        if is_circle:
            # 计算平均颜色进行最终确认
            mask = np.zeros_like(filtered_mask)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(img, mask=mask)
            
            # 检查是否为红色 (R > G and R > B)
            if mean_color[2] > mean_color[1] + 20 and mean_color[2] > mean_color[0] + 20:
                red_circles.append((int(x), int(y)))
    
    return np.array(red_circles, dtype=np.float32) if red_circles else None


def find_redpoint(img: np.ndarray, file_name: str) -> np.ndarray:
    """提取红色定位点并返回透视变换矩阵"""
    # 使用增强版检测方法
    red_points = adaptive_red_detection(img)
    
    if red_points is None or len(red_points) < 4:
        print(f"Error: Found {len(red_points) if red_points else 0} red points")
        return None
    
    # 改进的点排序方法
    # 计算中心点
    center = np.mean(red_points, axis=0)
    
    # 按角度排序
    def sort_key(point):
        diff = point - center
        return np.arctan2(diff[1], diff[0])
    
    sorted_points = sorted(red_points, key=sort_key)
    
    # 取面积最大的4个点
    if len(sorted_points) > 4:
        # 按距离中心点距离排序（近似面积）
        sorted_points = sorted(sorted_points, 
                              key=lambda p: np.linalg.norm(p - center),
                              reverse=True)[:4]
    
    # 创建有序点数组 [左上, 右上, 右下, 左下]
    src_points = np.array(sorted_points, dtype=np.float32)
    
    # ======== DEBUG: 在原图上绘制定位点 ========
    debug_img = img.copy()
    # 绘制所有检测到的点（蓝色）
    for i, point in enumerate(red_points):
        cv2.circle(debug_img, tuple(map(int, point)), 20, (255, 0, 0), 5)  # 蓝色圆
    
    # 绘制最终选中的4个点（红色）并标记顺序
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, point in enumerate(src_points):
        cv2.circle(debug_img, tuple(map(int, point)), 30, colors[i], 8)
        cv2.putText(debug_img, str(i), tuple(map(int, point)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 5)
    
    # 绘制中心点（黄色）
    cv2.circle(debug_img, tuple(map(int, center)), 10, (0, 255, 255), -1)
    
    # 保存调试图像
    debug_output = os.path.join(DEBUG_PATH, f"{file_name}_redpoints.jpg")
    cv2.imwrite(debug_output, debug_img)
    print(f"DEBUG: 定位点检测结果已保存到 {debug_output}")
    # ======== END DEBUG ========
    
    # 定义目标点位置
    dst_points = np.array([
        [0, 0],          # 左上角
        [W - 1, 0],      # 右上角
        [W - 1, H - 1],  # 右下角
        [0, H - 1]       # 左下角
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    return cv2.getPerspectiveTransform(src_points, dst_points)

def detect_grid_lines(image, file_name: str):  # 修改函数签名
    """使用通用方法检测网格线位置（不依赖颜色）"""
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 边缘检测
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    
    # 形态学操作强化网格线
    kernel_h = np.ones((1, 30), np.uint8)  # 水平线结构元素
    kernel_v = np.ones((30, 1), np.uint8)  # 垂直线结构元素
    
    # 分别提取水平和垂直线
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
    
    # 投影法检测网格线位置
    h_proj = np.sum(horizontal, axis=1)  # 水平投影（行和）
    v_proj = np.sum(vertical, axis=0)    # 垂直投影（列和）
    
    # 自适应阈值
    h_thresh = np.max(h_proj) * 0.2
    v_thresh = np.max(v_proj) * 0.2
    
    # 寻找峰值
    h_peaks = np.where(h_proj > h_thresh)[0]
    v_peaks = np.where(v_proj > v_thresh)[0]
    
    # 聚类分组（合并邻近的线）
    def cluster_peaks(peaks, max_gap=10):
        if len(peaks) == 0:
            return []
        peaks = np.sort(peaks)
        clusters = []
        current = [peaks[0]]
        
        for p in peaks[1:]:
            if p - current[-1] <= max_gap:
                current.append(p)
            else:
                clusters.append(int(np.mean(current)))
                current = [p]
        clusters.append(int(np.mean(current)))
        return clusters
    
    h_lines = cluster_peaks(h_peaks)
    v_lines = cluster_peaks(v_peaks)
    
    # 过滤边缘附近的线
    h_lines = [y for y in h_lines if 50 < y < H - 50]
    v_lines = [x for x in v_lines if 50 < x < W - 50]
    
    # 添加边界线（0和H/W-1）以确保完整网格
    h_lines = [0] + sorted(h_lines) + [H - 1]
    v_lines = [0] + sorted(v_lines) + [W - 1]
    
    # ======== DEBUG: 显示分割线 ========
    debug_img = image.copy()
    # 绘制水平线 (红色)
    for y in h_lines:
        cv2.line(debug_img, (0, y), (W-1, y), (0, 0, 255), 3)
    
    # 绘制垂直线 (绿色)
    for x in v_lines:
        cv2.line(debug_img, (x, 0), (x, H-1), (0, 255, 0), 3)
    
    # 添加文本信息
    cv2.putText(debug_img, f"H: {len(h_lines)} lines", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(debug_img, f"V: {len(v_lines)} lines", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # 保存调试图像
    debug_output = os.path.join(DEBUG_PATH, f"{file_name}_gridlines.jpg")
    cv2.imwrite(debug_output, debug_img)
    print(f"DEBUG: 网格线检测结果已保存到 {debug_output}")
    # ======== END DEBUG ========
    
    return h_lines, v_lines

def matrix_transform(img: np.ndarray, ts_matrix: np.ndarray) -> np.ndarray:
    """将原图进行透视变换，返回图形"""
    # 应用透视变换
    warped = cv2.warpPerspective(img, ts_matrix, (W, H))
    return warped

def split_image_by_grid(image, h_lines, v_lines, file_name):
    """依据网格进行分割"""
    output_dir = os.path.join(OUTPUT_PATH, file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加图像边界作为网格线
    all_h_lines = [0] + sorted(h_lines) + [H - 1]
    all_v_lines = [0] + sorted(v_lines) + [W - 1]
    
    cell_count = 0
    for i in range(len(all_h_lines) - 1):
        for j in range(len(all_v_lines) - 1):
            y1 = all_h_lines[i]
            y2 = all_h_lines[i + 1]
            x1 = all_v_lines[j]
            x2 = all_v_lines[j + 1]
            
            # 确保单元格大小合理
            if (y2 - y1) < 10 or (x2 - x1) < 10:
                continue
                
            # 裁剪单元格
            cell = image[y1:y2, x1:x2]
            cell_path = os.path.join(output_dir, f"cell_{cell_count}.png")
            cv2.imwrite(cell_path, cell)
            cell_count += 1

    print(f"图像 {file_name} 分割完成，共 {cell_count} 个单元格")
    return cell_count

if __name__ == "__main__":
    image_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    png_file_ls = list(image_path.rglob("*.png"))
    if not png_file_ls:
        print("没有找到png文件，或路径错误")
    else:
        processed_count = 0
        print("共检测到{}个png文件".format(len(png_file_ls)))
        for png_file in png_file_ls:
            print(f"处理文件: {png_file.name}")
            original_img = cv2.imread(str(png_file))
            if original_img is None:
                print(f"无法读取图像: {png_file}")
                continue
                
            # 创建高饱和副本用于定位点检测
            saturated_img = enhance_saturation(original_img, saturation_factor=3.0)
            
            # 提取红色定位点 (传递文件名参数)
            ts_matrix = find_redpoint(saturated_img, png_file.stem)
            if ts_matrix is None:
                print(f"无法找到足够的红色定位点: {png_file.name}")
                continue
                
            # 使用原始图像进行变换
            try:
                standard_image = cv2.warpPerspective(original_img, ts_matrix, (W, H))
            except Exception as e:
                print(f"透视变换失败: {str(e)}")
                continue
                
            # 检测网格线 (传递文件名参数)
            h_lines, v_lines = detect_grid_lines(standard_image, png_file.stem)
            if not h_lines or not v_lines:
                print(f"无法检测网格线: {png_file.name}")
                continue
                
            # 使用原始标准图像进行分割
            cell_count = split_image_by_grid(standard_image, h_lines, v_lines, png_file.stem)
            if cell_count > 0:
                processed_count += 1
                
        print(f"处理完成! 共处理 {processed_count}/{len(png_file_ls)} 个文件")