# 🤖 Self-Improving AI Multi-Agent Workstation

A premium, all-in-one machine learning workstation that leverages **LLM Agents**, **Vector Memory**, and **Automated Pipelines** to build, tune, and critique ML models. This workstation doesn't just train models—it learns from its own failures and successes over time.

---

## 🌟 Key Features

### 🧠 Self-Improving Logic
- **Pinecone Vector Memory**: Stores past model results and AI critiques to avoid repeating mistakes.
- **Agentic Self-Critique**: A specialized AI Data Scientist analyzes every pipeline run and provides actionable improvement suggestions.

### 📈 Supervised ML Pipeline (10+ Models)
- **Wide Algorithm Support**: Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost, KNN, Decision Tree, MLP Neural Network, SVC/SVR, and Linear/Logistic Regression.
- **🔥 Advanced Auto-Tuning**: Uses `RandomizedSearchCV` with Stratified Cross-Validation to find the absolute best hyperparameters.
- **🛠️ Automated Preprocessing**: Includes Robust Scaling, Mean/Mode Imputation, Variance Threshold filtering, Correlation filtering, and **SMOTE** for class balancing.
- **🎯 Try the Model**: Interactive prediction form to test your best model on real-time inputs.

### 🧩 Unsupervised ML Pipeline
- **Auto-Clustering**: Automatically determines the optimal number of clusters (K) using Silhouette and Elbow methods.
- **Algorithm Leaderboard**: Compares KMeans, MiniBatchKMeans, Agglomerative, Gaussian Mixture, and DBSCAN to find the best grouping for your data.
- **✨ 3D Visuals**: Interactive PCA projections (2D and 3D) to visualize high-dimensional data distribution.

### 🎨 Premium UI/UX
- **Glassmorphism Design**: Modern, sleek Streamlit interface with a dark-mode optimized aesthetic.
- **Live AI Analyst**: Instant dataset overview and automated data cleaning with one-click CSV download.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- [Pinecone](https://www.pinecone.io/) Account (Free tier works great)
- [OpenRouter](https://openrouter.ai/) or OpenAI API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Krush2004/Self-Improving-ML-Pipeline.git
cd Self-Improving-ML-Pipeline

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=self-improving-agent-memory
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Main Streamlit interface and UI orchestration.
- `model/supervised_pipeline.py`: Advanced training, tuning, and evaluation logic.
- `model/unsupervised_pipeline.py`: Clustering algorithms and PCA visualization.
- `agent/agent.py`: LLM reasoning, memory retrieval, and self-critique logic.

---


## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
