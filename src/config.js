// Firebase Configuration
// This file can be generated from .env file using the generate_config.py script
// Or you can manually set these values

// Try to get from environment variables (if using Vite/React with env vars)
// Otherwise, these values will be used as defaults
const firebaseConfig = {
  apiKey: import.meta.env?.VITE_FIREBASE_API_KEY || 
          process.env?.REACT_APP_FIREBASE_API_KEY || 
          window?.env?.FIREBASE_API_KEY ||
          "AIzaSyAT2jJAMMXErx-IAErqw5uvHaEbiVTh_js",
  authDomain: import.meta.env?.VITE_FIREBASE_AUTH_DOMAIN || 
              process.env?.REACT_APP_FIREBASE_AUTH_DOMAIN || 
              window?.env?.FIREBASE_AUTH_DOMAIN ||
              "nba-props-app-57fec.firebaseapp.com",
  projectId: import.meta.env?.VITE_FIREBASE_PROJECT_ID || 
             process.env?.REACT_APP_FIREBASE_PROJECT_ID || 
             window?.env?.FIREBASE_PROJECT_ID ||
             "nba-props-app-57fec",
  storageBucket: import.meta.env?.VITE_FIREBASE_STORAGE_BUCKET || 
                 process.env?.REACT_APP_FIREBASE_STORAGE_BUCKET || 
                 window?.env?.FIREBASE_STORAGE_BUCKET ||
                 "nba-props-app-57fec.firebasestorage.app",
  messagingSenderId: import.meta.env?.VITE_FIREBASE_MESSAGING_SENDER_ID || 
                     process.env?.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || 
                     window?.env?.FIREBASE_MESSAGING_SENDER_ID ||
                     "139494696545",
  appId: import.meta.env?.VITE_FIREBASE_APP_ID || 
         process.env?.REACT_APP_FIREBASE_APP_ID || 
         window?.env?.FIREBASE_APP_ID ||
         "1:139494696545:web:004413270772fac564ac20",
  measurementId: import.meta.env?.VITE_FIREBASE_MEASUREMENT_ID || 
                 process.env?.REACT_APP_FIREBASE_MEASUREMENT_ID || 
                 window?.env?.FIREBASE_MEASUREMENT_ID ||
                 "G-XE7KWJBH0Z",
};

export default firebaseConfig;

