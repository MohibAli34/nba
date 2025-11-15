# 🏀 NBA Player Props Prediction Model

An interactive sports analytics web app built with **Python** and **Streamlit**, designed to generate **NBA player prop projections** with advanced matchup context, betting line analysis, and real-time data.  
Ideal for bettors, analysts, and sports data enthusiasts looking for fast, accurate, and transparent player insights.

👉 **Live Demo:** [https://nba-streamlit.akibrhast.synology.me](https://nba-streamlit.akibrhast.synology.me)

---

## ✨ Features

### 🕹️ 1. Smart Game Selection
- Auto-fetches upcoming NBA games (7-day window)
- Displays date, time, and team matchup
- Simple and intuitive workflow: Game ➝ Team ➝ Player ➝ Stat


### 📊 2. Player Projection Engine
- Predicts:
- Points (PTS)
- Rebounds (REB)
- Assists (AST)
- 3-Pointers Made (3PM)
- Points + Rebounds + Assists (PRA)
- Double-Double probability
- Season blending logic: intelligently weighs current vs. prior season
- Recent performance indicators: last 5 / last 10 game averages

### 🛡️ 3. Matchup & Defensive Analytics
- Position-specific defensive rankings (1–30 scale)
- 10+ opponent metrics:
- Points allowed, FG%, FT%, 3PM allowed, REB, AST, STL, BLK, TO, etc.
- Matchup difficulty labels:
- `Elite`, `Above Average`, `Average`, `Below Average`
- Auto flags favorable/unfavorable matchups

### 💰 4. FanDuel Odds API Integration
- Real-time line fetching (via [The Odds API](https://the-odds-api.com/))
- Smart fallback to season average if no line available
- Manual override option for user-entered lines
- Edge calculation with visual recommendations:
- ✅ Over
- ❌ Under
- ⚪ No clear edge

### 📈 5. Hit Rate Analysis
- Season hit rate vs line
- Last 5 games hit rate
- Last 10 games hit rate

### 🆚 6. Head-to-Head Trends
- Career performance vs opponent
- Trend detection (up/down)
- Recent head-to-head game logs (up to 3 seasons)

### 🧠 7. Contextual Factors
- Rest days tracker
- Back-to-back game flag
- Opponent info panel

### 📋 8. Game Log Display
- Combines current + prior season for 10 recent games
- Full stat lines: PTS, REB, AST, 3PM, MIN, FG%, PRA
- Season labels clearly marked

### 🖥️ 9. Clean UI/UX
- Responsive layout built with Streamlit
- Color-coded recommendations
- Collapsible sections
- Fast interactions with smart caching

---

## 🧰 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Data Sources:** [nba_api](https://github.com/swar/nba_api), [HashtagBasketball](https://hashtagbasketball.com), [The Odds API](https://the-odds-api.com)
- **ML Model:** Ridge Regression
- **Backend:** Python (pandas, numpy, scikit-learn, requests)
- **Caching:** Streamlit cache + local disk caching

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/nba-player-props.git
cd nba-player-props

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🔑 Environment Variables

All API keys and configuration should be stored in a `.env` file in the project root. See `.env.example` or create `.env` with the following variables:

| Variable                      | Description                                   | Example                                     |
|-------------------------------|-----------------------------------------------|---------------------------------------------|
| `ODDS_API_KEY`                | API key for The Odds API                      | `f6aac04a6ab847bab31a7db076ef89e8`          |
| `FIREBASE_API_KEY`            | Firebase API key (for React client)           | `AIzaSyAT2jJAMMXErx-IAErqw5uvHaEbiVTh_js`   |
| `FIREBASE_AUTH_DOMAIN`        | Firebase authentication domain                | `nba-props-app-57fec.firebaseapp.com`       |
| `FIREBASE_PROJECT_ID`         | Firebase project ID                           | `nba-props-app-57fec`                       |
| `FIREBASE_STORAGE_BUCKET`     | Firebase storage bucket                       | `nba-props-app-57fec.firebasestorage.app`   |
| `FIREBASE_MESSAGING_SENDER_ID`| Firebase messaging sender ID                  | `139494696545`                              |
| `FIREBASE_APP_ID`             | Firebase app ID                               | `1:139494696545:web:004413270772fac564ac20` |
| `FIREBASE_MEASUREMENT_ID`     | Firebase analytics measurement ID             | `G-XE7KWJBH0Z`                              |
| `FIREBASE_CREDENTIALS_FILE`   | Path to Firebase admin SDK credentials JSON   | `nba-props-app-57fec-firebase-adminsdk-*.json` |
| `CACHE_DIR`                   | Directory for caching data (optional)         | `./cache`                                   |

**Note:** The `.env` file is gitignored for security. Copy your API keys from your Firebase console and The Odds API dashboard into the `.env` file.

## 🧪 Usage Guide

1. **Set up environment variables:**
   - Create a `.env` file in the project root (see `.gitignore` for example format)
   - Add all required API keys (see Environment Variables section above)
   - For React app: Run `python generate_config.py` to generate `src/config.js` from `.env`
   
2. **Launch the app:**  
   ```bash
   streamlit run app.py
   ```
   
3. Select an upcoming game from the dropdown.  
4. Choose a team and player.  
5. Pick a stat category (PTS, REB, AST, 3PM, PRA, DD).  
6. View projections, defensive matchup, hit rate, and edge recommendations.  
7. Compare the model output with sportsbook lines for decision-making.

---

## 📦 App Structure

- `app.py` – Main Streamlit UI  
- `utils/`
  - `data_fetcher.py` – NBA API, scraping & odds fetching  
  - `features.py` – Feature engineering  
  - `model.py` – Ridge regression model  
- `requirements.txt` – Dependencies  
- `Dockerfile` – Docker build for deployment  
- `render.yaml` – Optional Render hosting config  
- `README.md` – Project documentation

---

## 🛡️ Disclaimer

This tool is for informational and research purposes only.

All betting decisions are the responsibility of the user.

Odds are sourced from third-party APIs and may not reflect live sportsbook lines.
