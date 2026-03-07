import requests
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 🔐 安全获取环境变量 =================
# 动态获取环境变量中的 Key，绝对不会硬编码在文件里
API_KEY = os.getenv("S2_API_KEY")
# 如果获取不到，或者获取到的是我们填的占位符，就当做没有 Key
if API_KEY == "你的_Semantic_Scholar_密钥" or not API_KEY:
    API_KEY = None
# =======================================================

# ================= ⚙️ 配置区域 =================
QUERY = "quantum physics"
DOWNLOAD_LIMIT = 1
SAVE_DIR = "physics_knowledge_base"
BATCH_SIZE = 100
MAX_WORKERS = 8   # 并发线程数（建议 6-10）
# ===============================================

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)[:150].strip()

def create_session():
    session = requests.Session()
    # 代理设置已经移交给了 Slurm 的 run.sh，这里不需要写死了
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })
    return session

def get_adaptive_headers():
    """极其重要的自适应 Header：有 Key 就用，没 Key 就白嫖"""
    headers = {}
    if API_KEY:
        print("🔑 检测到有效 API Key，使用高级通道...")
        headers['x-api-key'] = API_KEY
    else:
        print("🚶 未检测到有效 API Key，使用公共免费通道（请注意速率限制）...")
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
    """
    接收参数，并返回成功下载的论文列表
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    api_headers = get_adaptive_headers()
    api_session = create_session()
    
    downloaded_papers = [] 
    downloaded_count = 0
    offset = 0

    print(f"🚀 [Scout] 开始 API 检索：{query}，目标 {limit} 篇")

    while downloaded_count < limit:
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "offset": offset,
            "limit": 10, 
            "fields": "title,isOpenAccess,openAccessPdf,citationCount,year",
            "sort": "citationCount:desc"
        }

        r = api_session.get(search_url, params=params, headers=api_headers)
        time.sleep(1.05)

        if r.status_code != 200:
            print(f"❌ API 错误: {r.status_code} - {r.text}")
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

                title = paper.get("title", "Untitled")
                filename = sanitize_filename(title) + ".pdf"
                filepath = os.path.join(save_dir, filename)
                
                paper_record = {"title": title, "pdf_path": filepath}

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
                else:
                    print("❌ 下载失败")

        offset += 10

    return downloaded_papers 

# ================= 本地/服务器独立测试入口 =================
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    api_headers = get_adaptive_headers()
    api_session = create_session()

    downloaded_count = 0
    offset = 0

    print(f"🚀 开始任务：{DOWNLOAD_LIMIT} 篇")

    while downloaded_count < DOWNLOAD_LIMIT:
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": QUERY,
            "offset": offset,
            "limit": BATCH_SIZE,
            "fields": "title,isOpenAccess,openAccessPdf,citationCount,year",
            "sort": "citationCount:desc"
        }

        print(f"\n📡 API 请求 offset={offset}")
        r = api_session.get(search_url, params=params, headers=api_headers)
        time.sleep(1.05) 

        if r.status_code != 200:
            print(f"❌ API 错误: {r.status_code} - {r.text}")
            break

        data = r.json()
        papers = data.get("data", [])
        if not papers:
            break

        tasks = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for paper in papers:
                if downloaded_count >= DOWNLOAD_LIMIT:
                    break

                pdf_info = paper.get("openAccessPdf")
                if not pdf_info or not pdf_info.get("url"):
                    continue

                title = paper.get("title", "Untitled")
                filename = sanitize_filename(title) + ".pdf"
                filepath = os.path.join(SAVE_DIR, filename)

                if os.path.exists(filepath):
                    continue

                session = create_session()
                tasks.append(
                    executor.submit(download_pdf, session, pdf_info["url"], filepath)
                )

                downloaded_count += 1

            for future in as_completed(tasks):
                if future.result():
                    print("✅ 下载完成")
                else:
                    print("❌ 下载失败")

        offset += BATCH_SIZE

    print(f"\n🎉 完成，共下载 {downloaded_count} 篇")

if __name__ == "__main__":
    main()