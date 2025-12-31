import os
import base64
import logging
import sys
from typing import List, Optional, Any
from pathlib import Path

# Add current directory to path so imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from openai import OpenAI
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from config import API_CONFIG

logger = logging.getLogger(__name__)

class DeepSeekOCRParser(BaseReader):
    """
    Use DeepSeek-OCR (via SiliconFlow/OpenAI API) to parse PDF files.
    Renders PDF pages as images using PyMuPDF (fitz) and sends them to the VLM for transcription.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, max_pages: Optional[int] = None):
        self.api_key = api_key or API_CONFIG["siliconflow"]["api_key"]
        self.base_url = base_url or "https://api.siliconflow.cn/v1"
        self.model = model or API_CONFIG["siliconflow"].get("ocr_model", "deepseek-ai/DeepSeek-OCR")
        self.max_pages = max_pages
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def load_data(self, file: Path, extra_info: Optional[dict] = None) -> List[Document]:
        if not isinstance(file, Path):
            file = Path(file)
            
        logger.info(f"Parsing PDF with DeepSeek OCR ({self.model}): {file}")
        
        try:
            import fitz
            import io
            from PIL import Image, ImageFilter
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for DeepSeekOCRParser. Please install it with `pip install pymupdf`.")

        try:
            doc = fitz.open(file)
            documents = []
            full_text = ""
            
            total_pages = len(doc)
            pages_to_process = total_pages
            if self.max_pages and self.max_pages > 0:
                pages_to_process = min(total_pages, self.max_pages)
                
            logger.info(f"PDF has {total_pages} pages. Processing first {pages_to_process} pages.")
            
            for i in range(pages_to_process):
                page = doc.load_page(i)
                logger.info(f"Processing page {i+1}/{pages_to_process}...")
                
                pix = page.get_pixmap(dpi=190)
                image_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                mime_type = "image/jpeg"
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                page_bytes = buf.getvalue()
                
                # Debug: save image
                debug_dir = Path("debug_sent_images")
                debug_dir.mkdir(exist_ok=True)
                with open(debug_dir / f"page_{i+1}.png", "wb") as f:
                    f.write(page_bytes)
                
                base64_image = base64.b64encode(page_bytes).decode('utf-8')
                
                # Call API
                try:
                    logger.info(f"Sending page {i+1} image to OCR model...")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "仅输出中文正文与阿拉伯数字及中文标点；不要输出英文字母、尖括号、花括号、或符号序列；如果无法识别则输出空。"},
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
                        frequency_penalty=0.8,
                        presence_penalty=0.0
                    )
                    content = response.choices[0].message.content
                    try:
                        import re
                        content = re.sub(r"[\\{\\}<>]+", "", content)
                        raw_lines = content.splitlines()
                        cleaned_lines = []
                        for ln in raw_lines:
                            s = ln.strip()
                            if not s:
                                continue
                            low = s.lower()
                            if re.search(r"(beginarray|endarray|enddocument|endarrayright|\^\]|\^\))", low):
                                continue
                            if re.search(r"\b(array|hline|textbf|textit|textcolor|cdots)\b", low):
                                continue
                            if re.search(r"[图表]\s*[0-9一二三四五六七八九十]+", s):
                                continue
                            if re.search(r"(?:t\^){3,}|\]{2,}|\[\s*\]", s):
                                continue
                            total = len(s)
                            cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
                            digits = len(re.findall(r"\d", s))
                            punct = len(re.findall(r"[，。；、：！？—《》“”‘’（）\-\.,]", s))
                            ratio = (cjk + digits + punct) / max(total, 1)
                            if ratio < 0.35:
                                continue
                            if digits / max(total, 1) > 0.6:
                                continue
                            if len(re.findall(r"[,\-\|/]", s)) > 8:
                                continue
                            if re.search(r"(.)\1{6,}", s):
                                continue
                            if re.fullmatch(r"[\s\W]+", s):
                                continue
                            cleaned_lines.append(s)
                        content = "\n".join(cleaned_lines)
                    except Exception:
                        pass
                    try:
                        import re
                        cn = len(re.findall(r"[\\u4e00-\\u9fff]", content))
                        bad = re.search(r"(?:\\^\\]?|t\\^){3,}|beginarray|endarray|enddocument", content)
                        need_retry = (cn < 30) or bool(bad)
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
                                            {"type": "text", "text": "仅输出中文正文与阿拉伯数字及中文标点；不要输出英文字母、尖括号、花括号、或符号序列；如果无法识别则输出空。"},
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
                                frequency_penalty=0.8,
                                presence_penalty=0.0
                            )
                            c2 = response2.choices[0].message.content
                            try:
                                import re
                                c2 = re.sub(r"[\\{\\}<>]+", "", c2)
                                lines2 = [ln.strip() for ln in c2.splitlines() if ln.strip() and not re.fullmatch(r"[\\s\\W]+", ln.strip())]
                                c2 = "\n".join(lines2)
                            except Exception:
                                pass
                            try:
                                cn2 = len(re.findall(r"[\\u4e00-\\u9fff]", c2))
                                if cn2 > cn:
                                    content = c2
                            except Exception:
                                pass
                        except Exception:
                            pass
                    full_text += content + "\n\n"
                    logger.info(f"OCR success for page {i+1}")
                except Exception as e:
                    logger.error(f"OCR failed for page {i+1}: {e}")
            
            doc.close()
            
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
            
            return [Document(text=full_text, metadata=extra_info or {})]
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise e
