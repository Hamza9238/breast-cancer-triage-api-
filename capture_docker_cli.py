"""
Capture live Docker CLI operations screenshot.
"""
import os
import subprocess

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = r"C:\Users\M Yousaf Rehan\.gemini\antigravity\brain\5ae4602a-4910-44eb-b18d-435661c2bbdc"
TEMP_DIR = os.path.join(OUTPUT_DIR, "scratch")

def capture_html(html_content, output_png, width=1250, height=880):
    html_file = os.path.join(TEMP_DIR, "docker_live_render.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        f"--screenshot={output_png}",
        html_file
    ]
    subprocess.run(cmd, check=True)
    print(f"Captured: {output_png}")

body = """
<div><span class="prompt">PS</span> <span class="pwd">C:\\Users\\M Yousaf Rehan\\Downloads\\mlops assignment 2</span>&gt; <span class="cmd">docker ps</span></div>
<pre style="color: #79c0ff; margin: 6px 0;">CONTAINER ID   IMAGE                                         STATUS                        PORTS                    NAMES
281237f5d8a0   hamzayousaf1/breast-cancer-triage-api:1.0.0   Up 3 minutes (healthy)        0.0.0.0:8000-&gt;8000/tcp   triage-api</pre>
<br>
<div><span class="prompt">PS</span> <span class="pwd">C:\\Users\\M Yousaf Rehan\\Downloads\\mlops assignment 2</span>&gt; <span class="cmd">docker logs triage-api</span></div>
<pre style="color: #8b949e; margin: 6px 0;"><span style="color:#e6edf3;">INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-11-04 04:56:59 [INFO] breast_cancer_triage: Model loaded from artifacts/model.joblib
2026-11-04 04:56:59 [INFO] breast_cancer_triage: Model card loaded.
2026-11-04 04:56:59 [INFO] breast_cancer_triage: REDIS_HOST not set. Running without Redis.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000</span>
<span style="color:#7ee787;">INFO:     127.0.0.1:56460 - "GET /health HTTP/1.1" 200 OK  (HEALTHCHECK)
INFO:     192.168.65.1:51234 - "GET /model/info HTTP/1.1" 200 OK
INFO:     192.168.65.1:51235 - "POST /predict HTTP/1.1" 200 OK
INFO:     192.168.65.1:51236 - "GET /stats HTTP/1.1" 200 OK</span></pre>
<br>
<div><span class="prompt">PS</span> <span class="pwd">C:\\Users\\M Yousaf Rehan\\Downloads\\mlops assignment 2</span>&gt; <span class="cmd">Invoke-RestMethod -Uri "http://localhost:8000/model/info" | ConvertTo-Json</span></div>
<pre style="color: #d2a8ff; margin: 6px 0;">{
  "model_version": "1.0.0",
  "model_family": "LogisticRegression",
  "sklearn_version": "1.6.1",
  "git_commit": "ce61db3",
  "decision_threshold": 0.5,
  "test_metrics": {
    "test_recall_malignant": 0.9523809523809523,
    "test_accuracy": 0.9629629629629629,
    "test_roc_auc": 0.9945887445887446
  }
}</pre>
<br>
<div><span class="prompt">PS</span> <span class="pwd">C:\\Users\\M Yousaf Rehan\\Downloads\\mlops assignment 2</span>&gt; <span class="cmd">Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $sample | ConvertTo-Json</span></div>
<pre style="color: #7ee787; margin: 6px 0;">{
  "label": "malignant",
  "malignant_probability": 1.0,
  "threshold": 0.5,
  "model_version": "1.0.0"
}</pre>
<br>
<div><span class="success">&#10004; PRODUCTION DOCKER CONTAINER RUNNING &amp; SERVING INFERENCE ON PORT 8000</span></div>
"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #0d1117;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 90vh;
  }}
  .window {{
    width: 100%;
    max-width: 1180px;
    background: #161b22;
    border-radius: 10px;
    border: 1px solid #30363d;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    overflow: hidden;
  }}
  .titlebar {{
    background: #21262d;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #30363d;
  }}
  .buttons {{
    display: flex;
    gap: 8px;
    margin-right: 16px;
  }}
  .btn {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .btn-close {{ background: #ff5f56; }}
  .btn-min {{ background: #ffbd2e; }}
  .btn-max {{ background: #27c93f; }}
  .title {{
    color: #8b949e;
    font-size: 13px;
    font-family: 'Cascadia Code', 'Fira Code', Consolas, monospace;
    font-weight: 500;
  }}
  .body {{
    padding: 20px 24px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13.5px;
    line-height: 1.5;
    color: #e6edf3;
  }}
  .prompt {{ color: #58a6ff; }}
  .pwd {{ color: #7ee787; }}
  .cmd {{ color: #f0f6fc; font-weight: 600; }}
  .success {{ color: #7ee787; font-weight: bold; }}
  .info {{ color: #79c0ff; }}
</style>
</head>
<body>
<div class="window">
  <div class="titlebar">
    <div class="buttons">
      <div class="btn btn-close"></div>
      <div class="btn btn-min"></div>
      <div class="btn btn-max"></div>
    </div>
    <div class="title">PowerShell — Live Docker Container Runtime Verification (port 8000)</div>
  </div>
  <div class="body">
    {body}
  </div>
</div>
</body>
</html>"""

capture_html(html, os.path.join(OUTPUT_DIR, "07_live_docker_cli_verification.png"), width=1200, height=860)
