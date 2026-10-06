# FastAPI Web 外壳: Http 接口 / WebSocket 流式问答 / 静态资源 / 会话历史 / 跨域 / 健康探针, 底层对接级联 RAG + MySQL。

from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional, List, Dict, Any
import time
import re
from new_main import IntegratedQASystem
# 静态资源绝对路径从它取
from base.config import Config
# 走完整子模块路径; 不要掏 qa_system.logger(那是编排器内部零件)
from base.logger import logger


app = FastAPI(title="carRAG API", description="汽车售后智能问答(MySQL 直答 + RAG 多模态检索生成)")

# 只放行本机回环来源(前端与本应用同源提供)。**不用 `*`**: 否则任意网页跨源读走
#   /api/sessions 与 /api/history/<sid>(可枚举全部 session_id 的入口); 且带 Cookie 那支
#   是回显来源、比 `*` 更松(starlette 0.46.2 cors.py), 配合 allow_credentials 浏览器不再拦截。
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态目录从配置取**绝对路径**(原来写死 CWD 相对的 'static', 从别的目录启动会静默 404)。
conf = Config()
APP_STATIC_DIR = conf.APP_STATIC_DIR

# 手册页图 URL 前缀 -> 走下方专用路由, 不是 app.mount。
PAGE_URL_PREFIX = '/api/page'


def _page_payload(spec):
    """
    函数功能: 页身份 -> 前端要的那一条引用页描述。
    :param spec: 'source|page' 字符串(新协议) 或 裸 int(老数据, 按默认书补全)
    :return: {'source': 书号, 'page': 0基页号, 'label': 书页上印的页码, 'url': 页图地址}
    label = page + 1: 页图是 0 基的(meta 32 -> 磁盘 p032.png), 手册上印刷的页码 = 页号+1,
      放后端算一次, 前端不用做算术。url 不含文件名: 页号 -> p{page:03d}.png 的映射只在
      rag_qa/core/image_leg.py 的 page_image_path() 一处。
    """
    if isinstance(spec, int):
        source, page = conf.MM_DEFAULT_BOOK, int(spec)
    else:
        s = str(spec).strip()
        if '|' in s:
            source, page_s = s.split('|', 1)
            page = int(page_s.strip())
        else:
            source, page = conf.MM_DEFAULT_BOOK, int(s)
    return {'source': source, 'page': page, 'label': page + 1,
            'url': f'{PAGE_URL_PREFIX}/{source}/{page}'}

os.makedirs(APP_STATIC_DIR, exist_ok=True)

# 页图目录不在版本控制里(生成物), 缺了不能静默(每条带页码回答变破图); 启动时把每本书页数打进日志。
_total_pages = 0
if os.path.isdir(conf.MM_PAGE_IMAGE_DIR):
    for _sid in conf.MM_BOOKS:
        _pdir = conf.book_image_dir(_sid)
        if os.path.isdir(_pdir):
            _n = len([f for f in os.listdir(_pdir)
                      if f.startswith('p') and f.endswith('.png')])
            logger.info(f'页图目录[{_sid}]: {_pdir} ({_n} 页)')
            _total_pages += _n
    if not _total_pages:
        logger.warning(f'页图根目录 {conf.MM_PAGE_IMAGE_DIR} 存在但没有任何书的页图 —— '
                       f'回答下方的"依据第X页"缩略图会全部 404')
else:
    logger.warning(f'页图目录不存在: {conf.MM_PAGE_IMAGE_DIR} —— '
                   f'回答下方的"依据第X页"缩略图会全部 404')

# 全局实例化问答核心。
qa_system = IntegratedQASystem()


# 全局固定问候语 正则配置。
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是车小问，专注于为车主答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是车小问，你的汽车手册问答助手，致力于提供汽车售后相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！我是车小问，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答汽车手册相关的问题！有什么我可以帮你的？"
    }
]


# Pydantic 请求/响应模型。
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


# 静态资源挂载与页面路由。
app.mount('/static', StaticFiles(directory=APP_STATIC_DIR), name='static')

# 根路径 GET 接口 -> 打开首页。
@app.get("/")
async def read_root():
    # 锚到绝对路径常量: 否则从别的目录启动服务时根路径 500。
    return FileResponse(os.path.join(APP_STATIC_DIR, 'index.html'))


# 手册页图接口: 页号 -> 整页 PNG(回答下方"依据第X页"缩略图)。
@app.get("/api/page/{source}/{page}")
async def get_page_image(source: str, page: int):
    """
    :param source: 书号(与 meta/日志/DB 里的 cited_pages 同一口径)
    :param page: 0 基页号
    用路由而非 app.mount('/pages', StaticFiles): 页图**不可变**, 显式给一周缓存
      (StaticFiles 不发 Cache-Control, 每次刷新 304); 且不把"页号->文件名"映射推进线上协议;
      `page: int` 让 FastAPI 在函数体之前把非数字打成 422, 边界语义一处说完。
    """
    # 惰性导入: image_leg 导入闭包约 1s(pymilvus), 只在本次请求用到, 放模块顶端白占用。
    from rag_qa.core.image_leg import page_image_path

    if page < 0:
        raise HTTPException(status_code=404, detail=f"页号越界: {page}")
    try:
        path = page_image_path(source, page)
    except KeyError:
        # 未知书号 -> 当页图不存在处理, 别让拼错的书号变成 500
        raise HTTPException(status_code=404, detail=f"未知书号: {source}")
    if not os.path.isfile(path):
        # 404 而不是 500: 页图是生成物, 缺一页不该让前端拿到"服务出错"
        raise HTTPException(status_code=404, detail=f"页图不存在: {source}|{page}")
    return FileResponse(
        path, media_type='image/png',
        headers={'Cache-Control': 'public, max-age=604800, immutable'})


# 创建新会话接口。
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# 查询历史消息接口(向上翻页)。
@app.get("/api/history/{session_id}")
async def get_history(session_id: str, limit: int = 20, before_id: Optional[int] = None):
    """
    取一页会话历史。
    :param limit: 本页最多多少轮(1~100)
    :param before_id: 向上翻页的游标 —— 传上一页返回的最旧一轮的 id
    """
    # limit 校验放 try **外面**: HTTPException 也是 Exception, 放进 try 会被下面 except 抓走
    #   改写 500; limit<=0 会让 has_more 永远 True 翻不出东西, 前端照 has_more 循环空转。
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit 必须在 1~100 之间")
    try:
        page = qa_system.get_session_history(session_id, limit=limit, before_id=before_id)
        # 每行补 pages(带 label 与页图 url)。老记录该列为 NULL -> 空列表 -> 前端不渲染页码条。
        #   **这里不做拒绝判断**: conversations 表没有 source 列, 读端分不出拒答 ——
        #   "拒答不带页码"那道闸门在写端 new_main.answer(_pick_cited_pages)。
        for item in page["history"]:
            item['pages'] = [_page_payload(p) for p in item.get('cited_pages') or []]
        return {"session_id": session_id,
                "history": page["history"],
                "has_more": page["has_more"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 会话列表接口 -> 前端左侧"历史会话"面板。
@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """会话列表(最多 500 个), 按最后活动时间倒序。"""
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit 必须在 1~500 之间")
    try:
        return {"sessions": qa_system.list_sessions(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


# 清除历史消息接口。
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# 公共工具: 检测日常问候并返回模板回复。
def check_greeting(query: str) -> Optional[str]:
    # 带图不配文字时 query 是 None -> 兜底空串, 否则 .strip() 会 AttributeError
    query_text = (query or "").strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None




# 非流式问答 POST 接口, 一次性返回完整回复。
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # L1 缓存 -> L2 题库, 走编排器的短路查询。
    #   不要直调 qa_system.bm25_search: 那会绕过 L1(不读不写), 这条接口的缓存永远建不起来。
    #   L3 直答 / L4 生成慢且要流式, 不在这条非流式接口跑。
    result = qa_system.lookup_text_cached(request.query)
    if result is None:
        # 两层都没命中, 需要 RAG, 提示使用 WebSocket
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    return {
        "answer": result.answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# 流式问答 WebSocket 接口, 逐块推送实现打字机效果。
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data.get("query")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()
            # 引用页描述。三处 end 帧共用这一份。问候天然 []。
            pages_payload = []
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "pages": pages_payload,
                        "processing_time": time.time() - start_time
                    })
                break
            # query_with_meta 在**最后一块**多带 cited_pages —— 那是 answer() 归一好的值
            #  (见 new_main._pick_cited_pages), 这里不重做任何判断。
            collected_answer = ""
            for token, is_complete, extra in qa_system.query_with_meta(
                    query, session_id=session_id):
                if extra:  # 非收尾的每一块 extra 都是 None
                    pages_payload = [_page_payload(p) for p in extra.get('cited_pages') or []]
                collected_answer += token
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "pages": pages_payload,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "pages": pages_payload,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")

# 健康检查接口(运维探针)。
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 主程序入口。
if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    uvicorn.run("app:app", host=host, port=port, reload=False)