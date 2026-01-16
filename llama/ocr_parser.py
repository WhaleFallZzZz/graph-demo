import os
import base64
import logging
import sys
from typing import List, Optional, Any
from pathlib import Path

# 添加当前目录到路径以便导入工作
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from openai import OpenAI
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama.config import API_CONFIG
from llama.common import retry_on_failure

logger = logging.getLogger(__name__)

class DeepSeekOCRParser(BaseReader):
    """
    使用 DeepSeek-OCR（通过 SiliconFlow/OpenAI API）解析 PDF 文件
    使用 PyMuPDF (fitz) 将 PDF 页面渲染为图像，并发送到 VLM 进行转录
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, max_pages: Optional[int] = None):
        self.api_key = api_key or API_CONFIG["siliconflow"]["api_key"]
        self.base_url = base_url or "https://api.siliconflow.cn/v1"
        self.model = model or API_CONFIG["siliconflow"].get("ocr_model", "deepseek-ai/DeepSeek-OCR")
        self.max_pages = max_pages
        # 设置60秒超时
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)

    def load_data(self, file: Path, extra_info: Optional[dict] = None) -> List[Document]:
        if not isinstance(file, Path):
            file = Path(file)
            
        logger.info(f"Parsing PDF with DeepSeek OCR ({self.model}): {file}")
        
        # 提取文件名用于调试图片命名
        def extract_filename_from_url_or_path(file_url_or_path: str) -> str:
            """从URL或路径中提取文件名（不含扩展名）"""
            from urllib.parse import urlparse, unquote
            
            # 如果是URL
            if file_url_or_path.startswith('http://') or file_url_or_path.startswith('https://'):
                parsed = urlparse(file_url_or_path)
                # 获取路径的最后一部分（去掉查询参数和片段）
                path = parsed.path
                # 处理双斜杠等情况，规范化路径
                path = path.replace('//', '/')
                # 获取最后一部分
                filename = path.split('/')[-1] if path else ''
                # URL解码
                filename = unquote(filename)
                # 去掉扩展名
                if '.' in filename:
                    filename = filename.rsplit('.', 1)[0]
                return filename if filename else 'unknown'
            else:
                # 如果是本地路径
                filename = Path(file_url_or_path).stem
                return filename if filename else 'unknown'
        
        # 获取文件标识符用于命名
        file_identifier = None
        if extra_info and 'file_url' in extra_info:
            file_url = extra_info['file_url']
            file_identifier = extract_filename_from_url_or_path(file_url)
            logger.debug(f"从 extra_info 中获取 file_url: {file_url}, 提取的文件名: {file_identifier}")
        else:
            # 如果没有 file_url，从文件路径中提取
            file_identifier = extract_filename_from_url_or_path(str(file))
            logger.debug(f"从文件路径中提取文件名: {file_identifier}")
        
        try:
            import fitz
            import io
            from PIL import Image, ImageFilter
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for DeepSeekOCRParser. Please install it with `pip install pymupdf`.")

        try:
                # 首先检查页数
            doc = fitz.open(file)
            total_pages = len(doc)
            pages_to_process = total_pages
            if self.max_pages and self.max_pages > 0:
                pages_to_process = min(total_pages, self.max_pages)
            doc.close()
                
            logger.info(f"PDF has {total_pages} pages. Processing first {pages_to_process} pages using multi-threading.")
            
            # 多线程设置
            # 基于 CPU 核心的动态调整，但限制以避免过多的 API 调用
            # 目标资源利用率 < 80%
            cpu_count = os.cpu_count() or 4
            max_workers = min(int(cpu_count * 0.8*2), 10)
            max_workers = max(1, max_workers) # Ensure at least 1 worker
            logger.info(f"Using {max_workers} threads for OCR processing (CPU count: {cpu_count}).")
            
            results = {}
            
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(iterable, **kwargs):
                    return iterable

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_page, i, file, file_identifier): i 
                    for i in range(pages_to_process)
                }
                
                for future in tqdm(as_completed(futures), total=pages_to_process, desc="OCR Processing", unit="page"):
                    page_idx = futures[future]
                    try:
                        idx, content = future.result()
                        results[idx] = content
                        logger.info(f"Page {idx+1} processing completed.")
                    except Exception as exc:
                        logger.error(f"Page {page_idx+1} generated an exception: {exc}")
                        results[page_idx] = ""

            # 按顺序合并结果
            full_text = ""
            for i in range(pages_to_process):
                if i in results:
                    full_text += results[i] + "\n\n"
            
            if not full_text.strip():
                logger.warning("No text extracted from PDF via OCR.")
                return []
                 
            try:
                import re
                lines = [ln for ln in full_text.splitlines()]
                deduped = []
                prev = None
                count = 0
                for ln in lines:
                    if ln == prev:
                        count += 1
                        if count <= 1:
                            deduped.append(ln)
                    else:
                        prev = ln
                        count = 0
                        deduped.append(ln)
                deduped = [ln for ln in deduped if not re.search(r"(.)\\1{8,}", ln)]
                full_text = "\n".join(deduped)
            except Exception:
                pass
            
            # OCR 诊断日志：记录提取的文本统计信息
            total_chars = len(full_text)
            avg_chars_per_page = total_chars / pages_to_process if pages_to_process > 0 else 0
            logger.info(
                f"📊 OCR 提取统计: 总字符数={total_chars:,}, "
                f"处理页数={pages_to_process}/{total_pages}, "
                f"平均每页字符数={avg_chars_per_page:.0f}"
            )
            
            return [Document(text=full_text, metadata=extra_info or {})]
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise e

    @retry_on_failure(max_retries=3, delay=1.0, backoff_factor=2.0)
    def _call_ocr_api_with_retry(self, base64_image: str, mime_type: str) -> str:
        """调用 OCR API 并返回内容（带重试）"""
        return self._call_ocr_api(base64_image, mime_type)

    def _call_ocr_api(self, base64_image: str, mime_type: str) -> str:
        """调用 OCR API 并返回内容"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请识别图片中的内容，并将其转换为 Markdown 格式。请保留标题层级、表格结构和列表格式。对于复杂的医学公式或符号，请使用 LaTeX 格式。如果无法识别则输出空。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.0,
            top_p=0.1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content

    def _process_page(self, i: int, file_path: Path, file_identifier: str) -> tuple[int, str]:
        """Process a single page: Render -> OCR -> Clean"""
        import fitz
        import io
        from PIL import Image, ImageFilter
        import re
        
        try:
            # Open document locally in thread
            doc = fitz.open(file_path)
            page = doc.load_page(i)
            
            # Render page
            pix = page.get_pixmap(dpi=190)
            image_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            mime_type = "image/jpeg"
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            page_bytes = buf.getvalue()
            
            # Debug save - commented out for performance optimization
            # debug_dir = Path("debug_sent_images")
            # debug_dir.mkdir(exist_ok=True)
            # debug_filename = f"{file_identifier}_page_{i+1}.png"
            # with open(debug_dir / debug_filename, "wb") as f:
            #     f.write(page_bytes)
            
            base64_image = base64.b64encode(page_bytes).decode('utf-8')
            
            # Call API with retry
            logger.info(f"Sending page {i+1} image to OCR model...")
            
            try:
                content = self._call_ocr_api_with_retry(base64_image, mime_type)
            except Exception as api_error:
                logger.error(f"OCR API call failed for page {i+1}: {api_error}")
                raise api_error
            
            # Clean content
            try:
                # 移除可能的 markdown 代码块标记
                content = re.sub(r"^```(markdown)?\s*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content)
                
                # 简单的行处理
                raw_lines = content.splitlines()
                cleaned_lines = []
                for ln in raw_lines:
                    s = ln.strip()
                    if not s: continue
                    cleaned_lines.append(s)
                content = "\n".join(cleaned_lines)
            except Exception:
                pass
                
            # Retry logic
            try:
                # 如果内容过短，可能需要重试（对于图片页面）
                cn = len(re.findall(r"[\u4e00-\u9fff]", content))
                # 检查是否包含表格结构的特征（如Markdown表格的分隔线）
                has_table = bool(re.search(r"\|.*\|", content)) and bool(re.search(r"\|---", content))
                
                # 如果中文很少且没有表格结构，可能是识别失败
                need_retry = (cn < 30) and (not has_table)
            except Exception:
                need_retry = False
                
            if need_retry:
                try:
                    pix2 = page.get_pixmap(dpi=210)
                    image_bytes2 = pix2.tobytes("png")
                    img2 = Image.open(io.BytesIO(image_bytes2)).convert("RGB")
                    img2 = img2.filter(ImageFilter.GaussianBlur(radius=0.3))
                    buf2 = io.BytesIO()
                    img2.save(buf2, format="JPEG", quality=88, optimize=True)
                    page_bytes2 = buf2.getvalue()
                    base64_image2 = base64.b64encode(page_bytes2).decode("utf-8")
                    
                    response2 = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "请识别图片中的内容，并将其转换为 Markdown 格式。请保留标题层级、表格结构和列表格式。如果无法识别则输出空。"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=4096,
                        temperature=0.0,
                        top_p=0.1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    c2 = response2.choices[0].message.content
                    try:
                        c2 = re.sub(r"^```(markdown)?\s*", "", c2, flags=re.IGNORECASE)
                        c2 = re.sub(r"\s*```$", "", c2)
                        lines2 = [ln.strip() for ln in c2.splitlines() if ln.strip()]
                        c2 = "\n".join(lines2)
                    except Exception:
                        pass
                    
                    # 如果重试结果看起来内容更多，则采用
                    if len(c2) > len(content):
                        content = c2
                except Exception:
                    pass
            
            doc.close()
            return i, content
        except Exception as e:
            logger.error(f"Error processing page {i+1}: {e}")
            # Ensure doc is closed
            try:
                doc.close()
            except:
                pass
            raise e
