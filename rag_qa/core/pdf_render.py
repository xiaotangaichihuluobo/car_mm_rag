# 手册 PDF 整页渲染 -> PNG: 建库第一步(页图既是检索键又是答案证据)。
# 位置: PDF -> 本文件 -> colqwen2_encoder -> image_store 入库

import os
import sys

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base.logger import logger

# 页图渲染的长边像素: 与建库冒烟验证时保持一致 = 1500。
#   改这个值会让新渲染的页图与冒烟验证过的页向量分布不一致, 非必要别动。
PAGE_RENDER_MAX_SIDE = 1500


def render_pdf_pages(pdf_path, output_dir, max_side=PAGE_RENDER_MAX_SIDE, force=False):
    """
    把 PDF 每一页渲染成一张 PNG(整页视觉检索的原料)。

    :param pdf_path:   PDF 绝对路径
    :param output_dir: PNG 输出目录(不存在则创建)
    :param max_side:   渲染长边像素; 按 max_side/页面长边 算缩放比例
    :param force:      已存在的 PNG 是否重新渲染。默认 False = 断点续跑,
                       编码中途中断后重跑不会白渲染 354 页
    :return: [{'page': 0基页号, 'png_path': 绝对路径}, ...] 按页号升序
    """
    import fitz      # PyMuPDF: 只在这一个函数里用, 故延迟导入, 不拖慢本模块其它用途

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = doc.page_count
    logger.info(f"开始渲染 PDF: {pdf_path} (共 {total} 页) -> {output_dir}")

    pages = []
    for page_no in range(total):
        png_path = os.path.join(output_dir, f"p{page_no:03d}.png")
        pages.append({'page': page_no, 'png_path': png_path})

        # 断点续跑: 已有页图就跳过渲染(354 页渲染一次要几分钟, 不该重复付这个代价)
        if not force and os.path.exists(png_path):
            continue

        page = doc[page_no]
        rect = page.rect
        # 缩放比例 = 目标长边 / 当前长边; 用 max 保证是"长边"对齐
        zoom = max_side / max(rect.width, rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(png_path)

        if (page_no + 1) % 50 == 0:
            logger.info(f"  已渲染 {page_no + 1}/{total} 页")

    doc.close()
    logger.info(f"页图渲染完成: {total} 页 -> {output_dir}")
    return pages
