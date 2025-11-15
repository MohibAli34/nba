# 🚀 Streamlit Cloud Deployment Guide

This guide explains how to deploy the NBA Player Props Prediction Model on Streamlit Cloud.

## 📋 Prerequisites

1. GitHub account with the repository pushed
2. Streamlit Cloud account (free tier available)
3. All API keys ready

## 🔧 Step-by-Step Deployment

### 1. Push Code to GitHub

Make sure your code is pushed to GitHub (already done ✅).

### 2. Set Up Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `MohibAli34/nba`
5. Set main file path: `app.py`
6. Click "Deploy"

### 3. Configure Environment Variables

**IMPORTANT:** Streamlit Cloud does NOT read `.env` files. You must set all environment variables in the Streamlit Cloud dashboard.

#### Required Environment Variables:

Go to your app settings → "Secrets" tab and add:

```toml
# Odds API
ODDS_API_KEY = "f6aac04a6ab847bab31a7db076ef89e8"

# Firebase Configuration (for React client if used)
FIREBASE_API_KEY = "AIzaSyAT2jJAMMXErx-IAErqw5uvHaEbiVTh_js"
FIREBASE_AUTH_DOMAIN = "nba-props-app-57fec.firebaseapp.com"
FIREBASE_PROJECT_ID = "nba-props-app-57fec"
FIREBASE_STORAGE_BUCKET = "nba-props-app-57fec.firebasestorage.app"
FIREBASE_MESSAGING_SENDER_ID = "139494696545"
FIREBASE_APP_ID = "1:139494696545:web:004413270772fac564ac20"
FIREBASE_MEASUREMENT_ID = "G-XE7KWJBH0Z"

# Firebase Admin SDK (for Python backend)
# Option 1: JSON string (recommended for Streamlit Cloud)
FIREBASE_CREDENTIALS_JSON = '{"type": "service_account", "project_id": "...", ...}'

# Option 2: File path (not recommended for Streamlit Cloud)
# FIREBASE_CREDENTIALS_FILE = "nba-props-app-57fec-firebase-adminsdk-*.json"
```

#### How to Get Firebase Credentials JSON:

1. Go to Firebase Console → Project Settings → Service Accounts
2. Click "Generate new private key"
3. Download the JSON file
4. Copy the entire JSON content
5. Paste it as a single-line string in `FIREBASE_CREDENTIALS_JSON` (remove newlines)

**Example:**
```toml
FIREBASE_CREDENTIALS_JSON = '{"type":"service_account","project_id":"nba-props-app-57fec",...}'
```

### 4. Verify Dependencies

Make sure `requirements.txt` includes all dependencies:
- ✅ Already includes `python-dotenv`
- ✅ Already includes `firebase-admin`
- ✅ All other dependencies are listed

### 5. Optional: Firebase Credentials File

If you prefer using the JSON file instead of environment variable:

1. Add the Firebase credentials JSON file to your repository (rename it to avoid `.gitignore`)
2. Set `FIREBASE_CREDENTIALS_FILE` environment variable to the filename
3. **Note:** This is less secure and not recommended for public repos

### 6. Test Deployment

After deployment:
1. Check the app logs for any errors
2. Test that the app loads
3. Verify API calls work (check Network tab in browser console)

## 🔍 Troubleshooting

### Issue: "ODDS_API_KEY not found"
**Solution:** Add `ODDS_API_KEY` to Streamlit Cloud Secrets

### Issue: "Firebase credentials file not found"
**Solution:** Add `FIREBASE_CREDENTIALS_JSON` as a JSON string in Streamlit Cloud Secrets, OR upload the credentials file and set `FIREBASE_CREDENTIALS_FILE`

### Issue: "Database initialization issue"
**Solution:** This is usually fine - the database will be created automatically. If it persists, check file permissions.

### Issue: App stuck on "Checking cache for game data..."
**Solution:** 
- Check if API keys are set correctly
- Check Streamlit Cloud logs for errors
- Try disabling Firebase cache temporarily (the app should still work)

### Issue: Module not found errors
**Solution:** Check that all dependencies are in `requirements.txt` and the app has been redeployed.

## 🔐 Security Notes

1. **Never commit `.env` file** - it's already in `.gitignore`
2. **Never commit Firebase credentials JSON** - use environment variables
3. **Use Streamlit Cloud Secrets** for all sensitive data
4. **Rotate API keys** if they're ever exposed

## 📝 Environment Variables Summary

| Variable | Required | Description |
|----------|----------|-------------|
| `ODDS_API_KEY` | ✅ Yes | The Odds API key |
| `FIREBASE_CREDENTIALS_JSON` | ⚠️ Optional | Firebase Admin SDK credentials (JSON string) |
| `FIREBASE_CREDENTIALS_FILE` | ⚠️ Optional | Firebase Admin SDK credentials (file path) |
| Other Firebase vars | ⚠️ Optional | Only needed if using React client |

The app will work without Firebase (with degraded caching). The only required variable is `ODDS_API_KEY` for odds integration.

## ✅ Checklist

- [ ] Code pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] `ODDS_API_KEY` set in Secrets
- [ ] `FIREBASE_CREDENTIALS_JSON` set in Secrets (if using Firebase)
- [ ] App loads without errors
- [ ] API calls work correctly
- [ ] Database initializes successfully

## 🆘 Need Help?

Check Streamlit Cloud logs in your app dashboard for detailed error messages.

