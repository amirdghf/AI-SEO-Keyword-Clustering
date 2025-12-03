import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ API KEYS CONFIGURATION
# ==========================================
# 🔴 کلید خود را اینجا بگذارید
GOOGLE_API_KEY = "AIzaSyBbfnf3knPoAxf6wKcdKEjaIG4ZAf7PCkc" 
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxx" 

# File Names
INPUT_FILE = "keywords_input.xlsx"
OUTPUT_EXCEL = "keywords_final_strategy.xlsx"
OUTPUT_HTML = "interactive_topic_map.html"

print("========================================")
print("   SEO MASTER TOOL (BERT + GEMINI 2.0)  ")
print("========================================")

# ---------------------------------------------------------
# Step 0: Select AI
# ---------------------------------------------------------
print("\nSelect Analysis Engine:")
print("1. Google Gemini 2.0 (Recommended: Free & Smart)")
print("2. OpenAI ChatGPT")

AI_PROVIDER = ""
gpt_client = None
gemini_model = None

while True:
    user_choice = input("\n>>> Enter 1 or 2: ").strip()
    
    if user_choice == "1":
        AI_PROVIDER = "gemini"
        if "AIza" not in GOOGLE_API_KEY:
            print("❌ Error: GOOGLE_API_KEY is missing.")
            sys.exit()
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            # استفاده از مدل جدید و سریع فلش 2
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print(">>> 🔌 Connected to Google Gemini 2.0")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit()
        break

    elif user_choice == "2":
        AI_PROVIDER = "openai"
        try:
            from openai import OpenAI
            if "sk-" not in OPENAI_API_KEY:
                 print("❌ Error: OPENAI_API_KEY is missing.")
                 sys.exit()
            gpt_client = OpenAI(api_key=OPENAI_API_KEY)
            print(">>> 🔌 Connected to OpenAI.")
        except:
            print("❌ Error: openai module not found.")
            sys.exit()
        break
    else:
        print("⚠️ Invalid input.")

# ---------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------
if not os.path.exists(INPUT_FILE):
    print(f"⚠️ Input file not found.")
    sys.exit()

print(f"\n✅  Loading '{INPUT_FILE}'...")
df = pd.read_excel(INPUT_FILE)
if 'Search Volume' not in df.columns: df['Search Volume'] = 1
df['Search Volume'] = pd.to_numeric(df['Search Volume'], errors='coerce').fillna(0)
df = df.dropna(subset=['Keyword'])
keywords = df['Keyword'].tolist()
print(f">>> {len(keywords)} keywords loaded.")

# ---------------------------------------------------------
# Step 2: Vectorization (BERT - The Brain of Clustering)
# ---------------------------------------------------------
# اینجا کلید ماجراست: استفاده از BERT برای ریاضیات دقیق
print(">>> 🧠  Phase 1: High-Precision Clustering with BERT...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(keywords)

# ---------------------------------------------------------
# Step 3: Auto-Clustering
# ---------------------------------------------------------
print(">>> 🤖  Optimizing clusters structure...")
best_k = 2
# اگر تعداد کلمات زیاد بود، هوشمندانه تعداد خوشه را پیدا کن
if len(keywords) > 5:
    max_k = min(50, max(2, int(len(keywords) * 0.65)))
    best_score = -1
    print("    Calculating math...", end="")
    for k in range(2, max_k + 1):
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels_test = kmeans_test.fit_predict(embeddings)
        if len(set(labels_test)) > 1:
            score = silhouette_score(embeddings, labels_test)
            if score > best_score:
                best_score = score
                best_k = k
    print(f" Done. Optimal Clusters: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
kmeans.fit(embeddings)
df['Cluster_ID'] = kmeans.labels_

# ---------------------------------------------------------
# Step 4: AI Analysis (Gemini - The Brain of Strategy)
# ---------------------------------------------------------
print(f"\n>>> 🔮  Phase 2: Strategic Analysis with {AI_PROVIDER.upper()}...")

ai_results = {}
unique_clusters = sorted(df['Cluster_ID'].unique())

for cluster_id in unique_clusters:
    cluster_keywords = df[df['Cluster_ID'] == cluster_id]['Keyword'].tolist()
    # فرستادن کلمات بیشتر برای درک بهتر
    keywords_str = ", ".join(cluster_keywords[:20])
    
    # پرامپت بسیار دقیق برای جلوگیری از خطا
    prompt = f"""
    You are a Senior SEO Strategist. I have grouped these keywords together:
    [{keywords_str}]
    
    Task:
    1. Analyze the semantics and intent of these keywords.
    2. Create a short, perfect "Pillar Page Title" in Persian (Farsi).
    3. Identify the "Search Intent" (Transactional, Informational, Commercial, or Navigational).
    
    Output format (Strictly one line):
    Title | Intent
    
    Example:
    راهنمای جامع خرید موبایل | Transactional
    """
    
    print(f"    Analyzing Cluster {cluster_id + 1}/{len(unique_clusters)}...", end="\r")
    
    topic = f"Group {cluster_id}"
    intent = "Unknown"

    try:
        res_txt = ""
        if AI_PROVIDER == "gemini":
            response = gemini_model.generate_content(prompt)
            res_txt = response.text.strip()
            time.sleep(0.5) # وقفه کوتاه
        elif AI_PROVIDER == "openai":
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_txt = response.choices[0].message.content.strip()

        if "|" in res_txt:
            parts = res_txt.split("|")
            topic = parts[0].strip()
            intent = parts[1].strip()
        else:
            # پاکسازی خروجی‌های اضافی احتمالی
            topic = res_txt.replace('Title:', '').replace('Intent:', '').split('\n')[0]

    except Exception as e:
        print(f"\n⚠️ Analysis Error: {e}")

    ai_results[cluster_id] = {'Topic': topic, 'Intent': intent}

print("\n>>> ✅  Analysis Completed.")

# Map & Export
df['Pillar_Topic'] = df['Cluster_ID'].map(lambda x: ai_results[x]['Topic'])
df['Search_Intent'] = df['Cluster_ID'].map(lambda x: ai_results[x]['Intent'])

cluster_stats = df.groupby('Cluster_ID')['Search Volume'].sum().reset_index()
cluster_stats.rename(columns={'Search Volume': 'Cluster_Total_Volume'}, inplace=True)
df = df.merge(cluster_stats, on='Cluster_ID')
df = df.sort_values(by=['Cluster_Total_Volume', 'Search Volume'], ascending=[False, False])

cols = ['Cluster_ID', 'Pillar_Topic', 'Search_Intent', 'Cluster_Total_Volume', 'Keyword', 'Search Volume']
for c in cols: 
    if c not in df.columns: df[c] = ""

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"🎉  Excel Saved: {OUTPUT_EXCEL}")

# Plot
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]
# سایز حباب‌ها رو منطقی‌تر می‌کنیم
df['Size'] = np.log1p(df['Search Volume']) * 5 + 5

fig = px.scatter(
    df, x='x', y='y', color='Pillar_Topic', size='Size', size_max=50,
    hover_name='Keyword', 
    hover_data={'Pillar_Topic': True, 'Search_Intent': True, 'Search Volume': True},
    title=f'SEO Strategy Map (BERT Clustering + {AI_PROVIDER.upper()} Analysis)', 
    template='plotly_white'
)
fig.write_html(OUTPUT_HTML)
print(f"🎉  Map Saved: {OUTPUT_HTML}")

input("\nPress Enter to exit...")