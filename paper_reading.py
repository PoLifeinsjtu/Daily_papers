import requests
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv("S2_API_KEY")
if API_KEY == "你的_Semantic_Scholar_密钥" or not API_KEY:
    API_KEY = None

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)[:150].strip()

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })
    return session

def get_adaptive_headers():
    headers = {}
    if API_KEY:
        headers['x-api-key'] = API_KEY
    return headers

def download_pdf(session, url, filepath):
    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
                return True
    except Exception:
        pass
    return False

# ================= 给 LangGraph Agent 调用的接口 =================
def fetch_papers(query: str, limit: int = 5, save_dir: str = "physics_knowledge_base") -> list:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    api_headers = get_adaptive_headers()
    api_session = create_session()
    
    downloaded_papers = [] 
    downloaded_count = 0
    offset = 0

    print(f"🚀 [Scout] 开始 API 检索最新文献：{query}，目标 {limit} 篇")

    while downloaded_count < limit:
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "offset": offset,
            "limit": 10, 
            "fields": "paperId,title,isOpenAccess,openAccessPdf,publicationDate", # 新增 paperId
            "sort": "publicationDate:desc" # 🌟 核心升级：按最新日期排序，不再看老旧的高引文
        }

        r = api_session.get(search_url, params=params, headers=api_headers)
        time.sleep(1.05)

        if r.status_code != 200:
            print(f"❌ API 错误: {r.status_code}")
            break

        papers = r.json().get("data", [])
        if not papers:
            break

        tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for paper in papers:
                if downloaded_count >= limit:
                    break

                pdf_info = paper.get("openAccessPdf")
                if not pdf_info or not pdf_info.get("url"):
                    continue

                paper_id = paper.get("paperId", "unknown_id")
                title = paper.get("title", "Untitled")
                filename = sanitize_filename(title) + ".pdf"
                filepath = os.path.join(save_dir, filename)
                
                # 🌟 核心升级：记录 paperId 供去重使用
                paper_record = {"paperId": paper_id, "title": title, "pdf_path": filepath}

                if os.path.exists(filepath):
                    downloaded_papers.append(paper_record)
                    downloaded_count += 1
                    continue

                session = create_session()
                tasks.append(
                    (executor.submit(download_pdf, session, pdf_info["url"], filepath), paper_record)
                )

            for future, record in tasks:
                if future.result():
                    downloaded_papers.append(record)
                    downloaded_count += 1
                    print(f"✅ 下载完成: {record['title'][:30]}...")

        offset += 10

    return downloaded_papers