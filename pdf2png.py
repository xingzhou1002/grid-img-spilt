import fitz
from pathlib import Path
import sys


def pdf2png(pdf):
    """
    输入工作目录下的文件名，带拓展名。只能为一个。
    可以转换多页的pdf文件。
    """
    i = 0
    files = pdf
    document = fitz.open("./" + files)  # 打开 PDF 文件
    try:
        for page in document:  # 遍历每一页
            pager = document.load_page(i)  # 加载第 i 页
            i += 1
            pix = pager.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # 获取图像
            output = f"./PNG_{files}_Page_{i}.png"  # 输出文件名
            pix.save(output)  # 保存图像
    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return

    print(f"Converted {i} pages from '{files}' to PNG images.")


def rpcpdf2png(wkpath: Path):
    """
    将路径下的所有 PDF 文件转换为 PNG 图像。
    """
    if not wkpath.exists() or not wkpath.is_dir():
        print(f"The provided path '{wkpath}' is not valid.")
        return

    for pdf_file in wkpath.glob("*.pdf"):  # 遍历路径下的所有 PDF 文件
        pdf2png(pdf_file.name)  # 调用 pdf2png 函数转换每个 PDF 文件


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf2png.py <path_to_pdf_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    path = Path(input_path)

    if path.is_file() and path.suffix.lower() == ".pdf":
        # 如果输入是一个 PDF 文件
        pdf2png(path.name)
    elif path.is_dir():
        # 如果输入是一个目录
        rpcpdf2png(path)
    else:
        print("Input must be a PDF file or a directory containing PDF files.")
        sys.exit(1)