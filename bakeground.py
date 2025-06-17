import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import os

# === 配置参数 ===
# 网格参数 - 必须与分割程序中的参数完全一致
GRID_SIZE_MM = 20          # 网格大小（毫米）
MARKER_SIZE_MM = 20        # 定位块尺寸（毫米）
BORDER_MARGIN_MM = 10      # 定位块到边缘的距离（毫米）
A4_WIDTH_MM = 210          # A4纸宽
A4_HEIGHT_MM = 297         # A4纸高
OUTPUT_PDF_NAME = "grid_with_markers.pdf"

# === 计算网格尺寸 ===
def calculate_grid_dimensions():
    """计算网格的行数和列数"""
    # 计算有效绘图区域
    effective_width = A4_WIDTH_MM - 2 * BORDER_MARGIN_MM
    effective_height = A4_HEIGHT_MM - 2 * BORDER_MARGIN_MM
    
    # 计算行列数
    cols = int(effective_width // GRID_SIZE_MM)
    rows = int(effective_height // GRID_SIZE_MM)
    
    # 计算实际使用的空间
    used_width = cols * GRID_SIZE_MM
    used_height = rows * GRID_SIZE_MM
    
    # 计算偏移量以使网格居中
    offset_x = (A4_WIDTH_MM - used_width) / 2
    offset_y = (A4_HEIGHT_MM - used_height) / 2
    
    return cols, rows, offset_x, offset_y

# === 生成带定位块的网格PDF ===
def generate_grid_pdf():
    """生成带定位块的网格PDF"""
    # 创建图形
    _fig, ax = plt.subplots(figsize=(A4_WIDTH_MM/25.4, A4_HEIGHT_MM/25.4), dpi=300)
    ax.set_xlim(0, A4_WIDTH_MM)
    ax.set_ylim(0, A4_HEIGHT_MM)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 反转Y轴使原点在左上角
    ax.set_axis_off()  # 隐藏坐标轴
    
    # 计算网格参数
    cols, rows, offset_x, offset_y = calculate_grid_dimensions()
    
    # 绘制网格线
    for i in range(cols + 1):
        x = offset_x + i * GRID_SIZE_MM
        plt.plot([x, x], [offset_y, offset_y + rows * GRID_SIZE_MM], 
                'k-', linewidth=1, alpha=0.9, color = "#386aff")
    
    for j in range(rows + 1):
        y = offset_y + j * GRID_SIZE_MM
        plt.plot([offset_x, offset_x + cols * GRID_SIZE_MM], [y, y], 
                'k-', linewidth=1, alpha=0.9, color = "#386aff")
    
    # 定位点
    marker_positions = [
        (BORDER_MARGIN_MM + 5, BORDER_MARGIN_MM + 8),  # 左下角
        (A4_WIDTH_MM - BORDER_MARGIN_MM - MARKER_SIZE_MM + 15, BORDER_MARGIN_MM + 8),  # 右下角
        (BORDER_MARGIN_MM + 5, A4_HEIGHT_MM - BORDER_MARGIN_MM - MARKER_SIZE_MM + 12),  # 左上角
        (A4_WIDTH_MM - BORDER_MARGIN_MM - MARKER_SIZE_MM + 15, A4_HEIGHT_MM - BORDER_MARGIN_MM - MARKER_SIZE_MM + 12)  # 右上角
    ]
    for persion in marker_positions:
        point = patches.Circle(persion, 2, facecolor = "#ff3838", edgecolor = "#ff3838", zorder = 10)
        ax.add_patch(point)

    
   
    # 保存PDF
    with PdfPages(OUTPUT_PDF_NAME) as pdf:
        pdf.savefig(dpi=300, bbox_inches='tight', pad_inches=0)
    
    print(f"[+] 网格PDF已生成: {OUTPUT_PDF_NAME}")
    print(f"    - 网格尺寸: {cols}列 × {rows}行")
    print(f"    - 定位块尺寸: {MARKER_SIZE_MM}mm")
    print(f"    - 边缘距离: {BORDER_MARGIN_MM}mm")
    plt.close()

if __name__ == "__main__":
    generate_grid_pdf()