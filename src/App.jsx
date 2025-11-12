import React, { useState, useEffect, useMemo, useCallback } from "react";
import { motion, useScroll, useSpring } from "framer-motion";
// Firebase Imports (modular SDK)
import { initializeApp, getApp, getApps } from "firebase/app";
import {
  getFirestore,
  doc,
  getDoc,
  setDoc,
  Timestamp,
} from "firebase/firestore";
import { getAuth, signInAnonymously, onAuthStateChanged } from "firebase/auth";

// --- Firebase Config ---
const firebaseConfig = {
  apiKey: "AIzaSyAT2jJAMMXErx-IAErqw5uvHaEbiVTh_js",
  authDomain: "nba-props-app-57fec.firebaseapp.com",
  projectId: "nba-props-app-57fec",
  storageBucket: "nba-props-app-57fec.firebasestorage.app",
  messagingSenderId: "139494696545",
  appId: "1:139494696545:web:004413270772fac564ac20",
  measurementId: "G-XE7KWJBH0Z",
};

// --- Initialize Firebase ---
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
export const db = getFirestore(app);
export const auth = getAuth(app);

// --- Auth Handling ---
onAuthStateChanged(auth, (user) => {
  if (!user) {
    console.log("Firebase: No user, attempting to sign in.");
    signInAnonymously(auth)
      .then(() => console.log("Signed in anonymously"))
      .catch((err) => console.error("Firebase Auth Error:", err));
  } else {
    console.log("Firebase user ready:", user.uid);
  }
});

// --- SVG Icons ---
const SunIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={2}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M12 12a5 5 0 100-10 5 5 0 000 10z"
    />
  </svg>
);
const MoonIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={2}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
    />
  </svg>
);
const StarIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 20 20"
    fill="currentColor"
  >
    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
  </svg>
);
const PlusIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={3}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
  </svg>
);
const MinusIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={3}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M18 12H6" />
  </svg>
);
const SearchIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={2}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
    />
  </svg>
);
const CloseIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth={2}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);
const BasketballLogo = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 100 100"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <circle cx="50" cy="50" r="48" fill="currentColor" />
    <path
      d="M50 2C50 2 70 25 70 50C70 75 50 98 50 98"
      stroke="#1F2937"
      strokeWidth="2.5"
    />
    <path
      d="M50 2C50 2 30 25 30 50C30 75 50 98 50 98"
      stroke="#1F2937"
      strokeWidth="2.5"
    />
    <path
      d="M2 50C2 50 25 30 50 30C75 30 98 50 98 50"
      stroke="#1F2937"
      strokeWidth="2.5"
    />
    <path
      d="M2 50C2 50 25 70 50 70C75 70 98 50 98 50"
      stroke="#1F2937"
      strokeWidth="2.5"
    />
  </svg>
);

// --- Date Helpers ---
// Get "2025-11-06" as the cache key
const getTodayCacheKey = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = (today.getMonth() + 1).toString().padStart(2, "0");
  const day = today.getDate().toString().padStart(2, "0");
  return `${year}-${month}-${day}`;
};

const getShortDate = (date) => {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
};

// --- LOADING SCREEN COMPONENT ---
function LoadingScreen({ onLoaded }) {
  const [quoteVisible, setQuoteVisible] = useState(false);

  useEffect(() => {
    const quoteTimer = setTimeout(() => setQuoteVisible(true), 500);
    const loadTimer = setTimeout(() => onLoaded(), 2000);

    return () => {
      clearTimeout(quoteTimer);
      clearTimeout(loadTimer);
    };
  }, [onLoaded]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900"
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.5 } }}
    >
      <div className="flex flex-col items-center justify-center text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{
            opacity: 1,
            scale: 1,
            transition: { duration: 0.5, delay: 0.2 },
          }}
        >
          <BasketballLogo
            className="w-24 h-24 text-orange-500 animate-spin"
            style={{ animationDuration: "3s" }}
          />
        </motion.div>

        {quoteVisible && (
          <motion.p
            className="mt-6 text-lg font-medium text-gray-700 dark:text-gray-300 italic max-w-xs text-center"
            initial={{ opacity: 0, y: 10 }}
            animate={{
              opacity: 1,
              y: 0,
              transition: { duration: 0.5, delay: 0.5 },
            }}
          >
            "Success is not an accident, success is actually a choice."
          </motion.p>
        )}
      </div>
    </motion.div>
  );
}

// --- Components ---

/**
 * Header Component
 */
function Header({ page, setPage, darkMode, toggleDarkMode, betCount }) {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  const NavButton = ({ targetPage, children }) => (
    <button
      onClick={() => setPage(targetPage)}
      className={`px-4 py-2 rounded-md font-semibold transition-colors
        ${
          page === targetPage
            ? "bg-blue-600 text-white"
            : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"
        }
      `}
    >
      {children}
    </button>
  );

  return (
    <header className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-40">
      <nav className="container mx-auto px-4 py-3 flex justify-between items-center">
        <h1 className="text-xl md:text-2xl font-bold text-gray-900 dark:text-white">
          NBA Props Dashboard
        </h1>
        <div className="flex items-center space-x-2 md:space-x-4">
          <NavButton targetPage="home">HOME</NavButton>
          <div className="relative">
            <NavButton targetPage="betSheet">BET SHEET</NavButton>
            {betCount > 0 && (
              <motion.span
                className="absolute -top-2 -right-2 bg-red-600 text-white text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 15 }}
              >
                {betCount}
              </motion.span>
            )}
          </div>
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full text-gray-700 dark:text-gray-300 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            aria-label="Toggle dark mode"
          >
            {darkMode ? (
              <SunIcon className="w-6 h-6" />
            ) : (
              <MoonIcon className="w-6 h-6" />
            )}
          </button>
        </div>
      </nav>
      {/* Scroll Progress Bar */}
      <motion.div className="h-1 bg-blue-600 origin-left" style={{ scaleX }} />
    </header>
  );
}

/**
 * Game Filter Component
 */
function GameFilter({ allPlayers, selectedGameId, setSelectedGameId }) {
  const [gamesByDate, setGamesByDate] = useState({});
  const [dateLabels, setDateLabels] = useState({});

  useEffect(() => {
    // 1. Derive unique games
    const gameMap = new Map();
    allPlayers.forEach((p) => {
      if (!p.gameId || !p.gameDate || !p.gameDescription) return; // Skip players with missing data
      if (!gameMap.has(p.gameId)) {
        gameMap.set(p.gameId, {
          id: p.gameId,
          date: p.gameDate.split("T")[0], // Normalize date to YYYY-MM-DD
          description: p.gameDescription,
        });
      }
    });

    const sortedGames = Array.from(gameMap.values()).sort((a, b) => {
      try {
        return new Date(a.date) - new Date(b.date);
      } catch (e) {
        console.error("Invalid date for sorting:", a.date, b.date);
        return 0;
      }
    });

    // 2. Group games by date
    const groups = {};
    sortedGames.forEach((game) => {
      if (!groups[game.date]) {
        groups[game.date] = [];
      }
      groups[game.date].push(game);
    });
    setGamesByDate(groups);

    // 3. Create date labels
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    const todayISO = today.toISOString().split("T")[0];
    const tomorrowISO = tomorrow.toISOString().split("T")[0];

    const labels = {};
    Object.keys(groups).forEach((dateISO) => {
      if (dateISO === todayISO) {
        labels[dateISO] = "Today";
      } else if (dateISO === tomorrowISO) {
        labels[dateISO] = "Tomorrow";
      } else {
        // Use T12:00:00Z to avoid timezone issues when creating the date object
        labels[dateISO] = getShortDate(new Date(dateISO + "T12:00:00Z"));
      }
    });
    setDateLabels(labels);
  }, [allPlayers]);

  const GameButton = ({ gameId, children }) => (
    <button
      onClick={() => setSelectedGameId(gameId)}
      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
        selectedGameId === gameId
          ? "bg-blue-600 text-white"
          : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"
      }`}
    >
      {children}
    </button>
  );

  return (
    <motion.div
      className="my-4 p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <GameButton gameId="all">All Games</GameButton>
      <div className="flex flex-col gap-y-4">
        {Object.entries(gamesByDate).map(([date, games]) => (
          <div
            key={date}
            className="flex flex-col md:flex-row md:items-start space-y-2 md:space-y-0 md:space-x-2"
          >
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 w-16 flex-shrink-0 pt-1.5">
              {dateLabels[date] || "Future"}
            </h3>
            <div className="flex flex-wrap gap-2">
              {games.map((game) => (
                <GameButton key={game.id} gameId={game.id}>
                  {game.description}
                </GameButton>
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/**
 * Filters Component
 */
function Filters({
  searchTerm,
  setSearchTerm,
  hitRateFilter,
  setHitRateFilter,
}) {
  const hitRateOptions = ["L5", "L10", "Season"];

  return (
    <motion.div
      className="my-4 p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="flex flex-col md:flex-row md:items-center md:space-x-4 space-y-4 md:space-y-0">
        {/* Player Search */}
        <div className="relative flex-grow">
          <input
            type="text"
            placeholder="Search player name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <SearchIcon className="w-5 h-5 text-gray-400 dark:text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
        </div>

        {/* Hit Rate Buttons */}
        <div className="flex-shrink-0">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300 mr-2">
            Hit Rate:
          </span>
          <div className="inline-flex rounded-md shadow-sm" role="group">
            {hitRateOptions.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setHitRateFilter(option)}
                className={`px-4 py-2 text-sm font-medium transition-colors
                  ${
                    hitRateFilter === option
                      ? "bg-blue-600 text-white z-10 ring-2 ring-blue-500"
                      : "bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-600"
                  }
                  first:rounded-l-lg last:rounded-r-lg border border-gray-300 dark:border-gray-600 -ml-px focus:z-10 focus:outline-none
                `}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Player Row Component
 */
function PlayerRow({
  player,
  addPlayerToBetSheet,
  hitRateFilter,
  updatePlayerLine,
}) {
  const [manualLine, setManualLine] = useState("");

  const handleManualLineSet = () => {
    const lineValue = parseFloat(manualLine);
    if (!isNaN(lineValue)) {
      updatePlayerLine(player.id, lineValue);
    }
  };

  const line = player.bookLine;
  const projection = player.projection;
  const isOver = line !== null ? projection > line : null;
  const diff = line !== null ? (projection - line).toFixed(1) : null;

  return (
    <motion.tr
      className="border-b border-gray-200 dark:border-gray-700"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      whileHover={{ backgroundColor: "rgba(249, 250, 251, 1)" }} // Tailwind's bg-gray-50
      transition={{ duration: 0.3 }}
    >
      {/* Player */}
      <td className="px-4 py-3 whitespace-nowrap">
        <div className="flex items-center">
          <div className="font-medium text-gray-900 dark:text-white">
            {player.playerName}
          </div>
          {player.isStarter && (
            <StarIcon
              className="w-4 h-4 text-yellow-400 ml-1"
              title="Projected Starter"
            />
          )}
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {player.team} vs {player.opponent}
        </div>
      </td>
      {/* Stat */}
      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap">
        {player.stat}
      </td>
      {/* Projection */}
      <td className="px-4 py-3 text-sm text-center font-semibold text-gray-900 dark:text-white whitespace-nowrap">
        {projection.toFixed(1)}
      </td>

      {/* Book Line (Interactive) */}
      <td className="px-4 py-3 whitespace-nowrap">
        {line !== null ? (
          <div className="flex items-center justify-center space-x-2">
            <button
              onClick={() => updatePlayerLine(player.id, line - 0.5)}
              className="p-1 rounded-full bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500"
            >
              <MinusIcon className="w-4 h-4" />
            </button>
            <span className="text-sm font-medium w-10 text-center">
              {line.toFixed(1)}
            </span>
            <button
              onClick={() => updatePlayerLine(player.id, line + 0.5)}
              className="p-1 rounded-full bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
        ) : (
          // Manual Line Entry
          <div className="flex items-center space-x-1">
            <input
              type="number"
              step="0.5"
              placeholder="Set line"
              value={manualLine}
              onChange={(e) => setManualLine(e.target.value)}
              className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <button
              onClick={handleManualLineSet}
              className="px-2 py-1 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
            >
              Set
            </button>
          </div>
        )}
      </td>

      {/* Over/Under */}
      <td className="px-4 py-3 text-center whitespace-nowrap">
        {isOver !== null ? (
          <span
            className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
              isOver
                ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200"
                : "bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200"
            }`}
          >
            {isOver ? `O (${diff})` : `U (${diff})`}
          </span>
        ) : (
          <span className="text-sm text-gray-400">-</span>
        )}
      </td>

      {/* Hit Rate */}
      <td className="px-4 py-3 text-sm text-center text-gray-700 dark:text-gray-300 whitespace-nowrap">
        {player.hitRate[hitRateFilter]}
      </td>

      {/* Actions Cell */}
      <td className="px-4 py-3 text-center whitespace-nowrap space-x-2">
        <button
          onClick={() => addPlayerToBetSheet(player)}
          className="p-2 rounded-full bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-800"
          title="Add to Bet Sheet"
        >
          <PlusIcon className="w-5 h-5" />
        </button>
      </td>
    </motion.tr>
  );
}

/**
 * Player Table Component
 */
function PlayerTable({
  players,
  setPlayers,
  addPlayerToBetSheet,
  hitRateFilter,
  searchTerm,
  selectedGameId,
}) {
  const filteredPlayers = useMemo(() => {
    const gameFiltered = players.filter(
      (p) => selectedGameId === "all" || p.gameId === selectedGameId
    );
    const searchFiltered = gameFiltered.filter((p) =>
      p.playerName.toLowerCase().includes(searchTerm.toLowerCase())
    );
    // Sort logic from your original app: starters first, then by projection
    const sortedData = [...searchFiltered].sort((a, b) => {
      if (a.isStarter && !b.isStarter) return -1;
      if (!a.isStarter && b.isStarter) return 1;
      // Use a default projection of 0 if null/undefined
      const projA = a.projection || 0;
      const projB = b.projection || 0;
      return projB - projA;
    });
    return sortedData;
  }, [players, selectedGameId, searchTerm]);

  const updatePlayerLine = (playerId, newLine) => {
    setPlayers((currentPlayers) =>
      currentPlayers.map((p) =>
        p.id === playerId ? { ...p, bookLine: newLine } : p
      )
    );
  };

  const headers = [
    "Player",
    "Stat",
    "Projection",
    "Book Line",
    "O/U (Diff)",
    "Hit Rate",
    "Actions",
  ];

  return (
    <motion.div
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-x-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-700">
          <tr>
            {headers.map((header) => (
              <th
                key={header}
                scope="col"
                className={`px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider ${
                  [
                    "Projection",
                    "Book Line",
                    "O/U (Diff)",
                    "Hit Rate",
                    "Actions",
                  ].includes(header)
                    ? "text-center"
                    : ""
                }`}
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
          {filteredPlayers.length > 0 ? (
            filteredPlayers.map((player) => (
              <PlayerRow
                key={player.id}
                player={player}
                addPlayerToBetSheet={addPlayerToBetSheet}
                hitRateFilter={hitRateFilter}
                updatePlayerLine={updatePlayerLine}
              />
            ))
          ) : (
            <tr>
              <td
                colSpan={headers.length}
                className="px-4 py-6 text-center text-gray-500 dark:text-gray-400"
              >
                {selectedGameId === "all"
                  ? "No players found."
                  : "No players found for this game."}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </motion.div>
  );
}

/**
 * Home Page Component
 */
function HomePage({
  players,
  setPlayers,
  addPlayerToBetSheet,
  searchTerm,
  setSearchTerm,
  hitRateFilter,
  setHitRateFilter,
  loading,
  loadingProgress,
  selectedGameId,
  setSelectedGameId,
  gamesList,
  currentGameIndex,
  onLoadNextGame,
}) {
  const hasMoreGames = gamesList.length > 0 && currentGameIndex < gamesList.length - 1;
  const nextGame = hasMoreGames ? gamesList[currentGameIndex + 1] : null;

  return (
    <>
      <GameFilter
        allPlayers={players}
        selectedGameId={selectedGameId}
        setSelectedGameId={setSelectedGameId}
      />
      <Filters
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        hitRateFilter={hitRateFilter}
        setHitRateFilter={setHitRateFilter}
      />

      {loading ? (
        <motion.div
          className="flex flex-col justify-center items-center h-64"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <p className="ml-4 text-lg font-medium">
              Running models & fetching data... (This may take a minute)
            </p>
          </div>
          {loadingProgress && (
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              {loadingProgress}
            </p>
          )}
        </motion.div>
      ) : (
        <>
          <PlayerTable
            players={players}
            setPlayers={setPlayers}
            addPlayerToBetSheet={addPlayerToBetSheet}
            hitRateFilter={hitRateFilter}
            searchTerm={searchTerm}
            selectedGameId={selectedGameId}
          />
          
          {hasMoreGames && (
            <motion.div
              className="my-6 flex justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <button
                onClick={onLoadNextGame}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
              >
                <span>Search Next Game</span>
                {nextGame && (
                  <span className="text-sm opacity-90">
                    ({nextGame.game_description})
                  </span>
                )}
              </button>
            </motion.div>
          )}
          
          {!hasMoreGames && gamesList.length > 0 && (
            <motion.div
              className="my-6 flex justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                All games have been fetched! ({gamesList.length} total)
              </p>
            </motion.div>
          )}
        </>
      )}
    </>
  );
}

/**
 * Bet Sheet Page Component
 */
function BetSheetPage({ betSheet, removePlayerFromBetSheet }) {
  return (
    <motion.div
      className="my-4 p-4 md:p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
        My Bet Sheet
      </h2>
      {betSheet.length === 0 ? (
        <p className="text-gray-500 dark:text-gray-400">
          Your bet sheet is empty. Add players from the Home page.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Player
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Bet
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Line
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Projection
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Remove
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {betSheet.map((player) => {
                const isOver =
                  player.bookLine !== null &&
                  player.projection > player.bookLine;
                return (
                  <motion.tr
                    key={player.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    layout
                  >
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="font-medium text-gray-900 dark:text-white">
                        {player.playerName}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {player.team}
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                      {player.stat}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                      {player.bookLine !== null ? (
                        <span
                          className={
                            isOver
                              ? "text-green-600 dark:text-green-400"
                              : "text-red-600 dark:text-red-400"
                          }
                        >
                          {isOver ? "Over" : "Under"}{" "}
                          {player.bookLine.toFixed(1)}
                        </span>
                      ) : (
                        <span className="text-gray-400">N/A</span>
                      )}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                      {player.projection.toFixed(1)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-right">
                      <button
                        onClick={() => removePlayerFromBetSheet(player.id)}
                        className="p-1.5 rounded-full bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-800"
                        title="Remove from Bet Sheet"
                      >
                        <CloseIcon className="w-5 h-5" />
                      </button>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  );
}

/**
 * Main App Component
 */
export default function App() {
  // --- State ---
  const [page, setPage] = useState("home");
  const [darkMode, setDarkMode] = useState(false);
  const [appLoading, setAppLoading] = useState(true); // This is for the *initial* logo screen
  const [dataLoading, setDataLoading] = useState(true); // This is for the *table* loading spinner
  const [loadingProgress, setLoadingProgress] = useState(""); // Shows which game is being fetched
  const [allPlayers, setAllPlayers] = useState([]);
  const [gamesList, setGamesList] = useState([]); // Store the list of games
  const [currentGameIndex, setCurrentGameIndex] = useState(0); // Track which game we're on
  const [betSheet, setBetSheet] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [hitRateFilter, setHitRateFilter] = useState("L5");
  const [selectedGameId, setSelectedGameId] = useState("all");
  const [authReady, setAuthReady] = useState(false);
  const [error, setError] = useState(null); // For UI errors

  // --- Effects ---

  // Effect for Theme Toggle
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  // Effect for Firebase Auth
  useEffect(() => {
    if (!auth) {
      console.error("Firebase Auth is not initialized.");
      setError("App configuration error. Please contact support.");
      setAuthReady(true); // Set to true to stop blocking
      return;
    }
    const signIn = async () => {
      onAuthStateChanged(auth, (user) => {
        if (user) {
          console.log("Firebase: User is signed in.");
          setAuthReady(true);
        } else {
          console.log("Firebase: No user, attempting to sign in.");
          // This token is provided by the environment, if it exists
          const token =
            typeof __initial_auth_token !== "undefined"
              ? __initial_auth_token
              : null;
          if (token) {
            signInWithCustomToken(auth, token)
              .then(() => {
                console.log("Firebase: Signed in with custom token.");
                setAuthReady(true);
              })
              .catch((error) => {
                console.error("Firebase: Custom Auth Error:", error);
                // Fallback to anonymous
                signInAnonymously(auth).then(() => {
                  console.log("Firebase: Signed in anonymously as fallback.");
                  setAuthReady(true);
                });
              });
          } else {
            // Default to anonymous sign-in
            signInAnonymously(auth).then(() => {
              console.log("Firebase: Signed in anonymously.");
              setAuthReady(true);
            });
          }
        }
      });
    };
    signIn();
  }, []); // Empty dependency array means this runs once on mount

  // Function to fetch a single game by index
  const fetchGameData = useCallback(async (gameIndex) => {
    const cacheKey = getTodayCacheKey();
    const appId =
      typeof __app_id !== "undefined" ? __app_id : "default-app-id";
    
    const game = gamesList[gameIndex];
    if (!game) {
      console.log("Data Fetch: No game at index", gameIndex);
      return;
    }

    const progressMsg = `Fetching game ${gameIndex + 1}/${gamesList.length}: ${game.game_description}`;
    setLoadingProgress(progressMsg);
    console.log(`Data Fetch: ${progressMsg}`);

    try {
      const response = await fetch(`/api/get_data?game_index=${gameIndex}`);
      if (!response.ok) {
        const errData = await response
          .json()
          .catch(() => ({ error: "Unknown API error" }));
        throw new Error(
          `API Error: ${response.statusText} - ${errData.error || "Unknown"}`
        );
      }

      const gameData = await response.json();

      if (!Array.isArray(gameData)) {
        throw new Error("API did not return an array.");
      }

      // Add new players from this game to the existing list
      setAllPlayers((prevPlayers) => {
        const updatedPlayers = [...prevPlayers, ...gameData];
        
        // Save to Firestore cache (best-effort only)
        const cacheDocRef = doc(
          db,
          "artifacts",
          appId,
          "public/data/nba_props_cache",
          cacheKey
        );
        
        setDoc(cacheDocRef, {
          playerData: JSON.stringify(updatedPlayers),
          timestamp: Timestamp.now(),
        })
          .then(() => {
            console.log(`Data Fetch: Saved game ${gameIndex + 1} to cache.`);
          })
          .catch((cacheWriteErr) => {
            console.warn(
              "Data Fetch: Cache write failed. Proceeding without persisting cache.",
              cacheWriteErr
            );
          });

        return updatedPlayers;
      });

      console.log(
        `Data Fetch: Game ${gameIndex + 1} complete. Players: ${gameData.length}`
      );
      setLoadingProgress(""); // Clear progress message
    } catch (err) {
      console.error(`Failed to fetch game ${gameIndex + 1}:`, err);
      setError(
        `Failed to load game ${gameIndex + 1}: ${err.message}. Confirm the Python API at http://127.0.0.1:5000 is running.`
      );
      setLoadingProgress(""); // Clear progress message on error
    }
  }, [gamesList]);

  // Function to load next game
  const loadNextGame = useCallback(async () => {
    if (currentGameIndex >= gamesList.length - 1) {
      console.log("Data Fetch: All games have been fetched.");
      return;
    }

    setDataLoading(true);
    const nextIndex = currentGameIndex + 1;
    setCurrentGameIndex(nextIndex);
    await fetchGameData(nextIndex);
    setDataLoading(false);
  }, [currentGameIndex, gamesList.length, fetchGameData]);

  // Effect for Initial Data Fetch - Load first game only
  useEffect(() => {
    // Wait until Firebase Auth is ready before trying to read/write data
    if (!authReady || !db) {
      console.log("Data Fetch: Waiting for auth to be ready.");
      return;
    }

    const loadInitialData = async () => {
      console.log("Data Fetch: loadInitialData() started.");
      setDataLoading(true);
      setLoadingProgress(""); // Reset progress message
      setError(null);

      const cacheKey = getTodayCacheKey();
      const appId =
        typeof __app_id !== "undefined" ? __app_id : "default-app-id";
      // This is the public cache, shared by all users.
      const cacheDocRef = doc(
        db,
        "artifacts",
        appId,
        "public/data/nba_props_cache",
        cacheKey
      );

      try {
        // 1. Check Firestore Cache
        console.log(`Data Fetch: Checking cache at ${cacheDocRef.path}`);
        let cacheDoc = null;
        try {
          cacheDoc = await getDoc(cacheDocRef);
        } catch (cacheErr) {
          console.warn(
            "Data Fetch: Cache read failed. Continuing without Firestore cache.",
            cacheErr
          );
        }

        if (cacheDoc?.exists()) {
          console.log("Data Fetch: Cache HIT.");
          const cacheData = cacheDoc.data();
          // Check if cache is stale (e.g., older than 4 hours)
          const cacheAgeHours =
            (Timestamp.now().seconds - cacheData.timestamp.seconds) / 3600;

          if (cacheAgeHours < 4) {
            console.log("Data Fetch: Cache is fresh. Loading from Firestore.");
            setAllPlayers(JSON.parse(cacheData.playerData));
            setDataLoading(false);
            // Still fetch games list to show "Search Next Game" button if needed
            const gamesResponse = await fetch("/api/get_games");
            if (gamesResponse.ok) {
              const games = await gamesResponse.json();
              if (Array.isArray(games)) {
                setGamesList(games);
                // Set current index to last game if we have cached data
                setCurrentGameIndex(games.length - 1);
              }
            }
            return; // We are done, loaded from cache
          } else {
            console.log("Data Fetch: Cache is stale. Fetching new data.");
          }
        } else {
          console.log("Data Fetch: Cache MISS.");
        }

        // 2. Cache MISS or STALE: Fetch games list first
        console.log("Data Fetch: Fetching games list from /api/get_games");
        const gamesResponse = await fetch("/api/get_games");
        if (!gamesResponse.ok) {
          const errData = await gamesResponse
            .json()
            .catch(() => ({ error: "Unknown API error" }));
          throw new Error(
            `API Error: ${gamesResponse.statusText} - ${errData.error || "Unknown"}`
          );
        }
        const games = await gamesResponse.json();

        if (!Array.isArray(games)) {
          throw new Error("Games API did not return an array.");
        }

        if (games.length === 0) {
          console.log("Data Fetch: No games found.");
          setAllPlayers([]);
          setGamesList([]);
          setDataLoading(false);
          return;
        }

        setGamesList(games);
        setCurrentGameIndex(0);

        // 3. Fetch only the first game (use local games variable to avoid closure issue)
        console.log(`Data Fetch: Found ${games.length} games. Loading first game...`);
        const firstGame = games[0];
        if (firstGame) {
          const progressMsg = `Fetching game 1/${games.length}: ${firstGame.game_description}`;
          setLoadingProgress(progressMsg);
          console.log(`Data Fetch: ${progressMsg}`);

          try {
            const response = await fetch(`/api/get_data?game_index=0`);
            if (!response.ok) {
              const errData = await response
                .json()
                .catch(() => ({ error: "Unknown API error" }));
              throw new Error(
                `API Error: ${response.statusText} - ${errData.error || "Unknown"}`
              );
            }

            const gameData = await response.json();

            if (!Array.isArray(gameData)) {
              throw new Error("API did not return an array.");
            }

            // Add new players from this game to the existing list
            setAllPlayers((prevPlayers) => {
              const updatedPlayers = [...prevPlayers, ...gameData];
              
              // Save to Firestore cache (best-effort only)
              const cacheDocRef = doc(
                db,
                "artifacts",
                appId,
                "public/data/nba_props_cache",
                cacheKey
              );
              
              setDoc(cacheDocRef, {
                playerData: JSON.stringify(updatedPlayers),
                timestamp: Timestamp.now(),
              })
                .then(() => {
                  console.log(`Data Fetch: Saved game 1 to cache.`);
                })
                .catch((cacheWriteErr) => {
                  console.warn(
                    "Data Fetch: Cache write failed. Proceeding without persisting cache.",
                    cacheWriteErr
                  );
                });

              return updatedPlayers;
            });

            console.log(
              `Data Fetch: Game 1 complete. Players: ${gameData.length}`
            );
            setLoadingProgress(""); // Clear progress message
          } catch (err) {
            console.error(`Failed to fetch game 1:`, err);
            setError(
              `Failed to load game 1: ${err.message}. Confirm the Python API at http://127.0.0.1:5000 is running.`
            );
            setLoadingProgress(""); // Clear progress message on error
          }
        }
      } catch (err) {
        console.error("Failed to load initial data:", err);
        setError(
          `Failed to load data: ${err.message}. Confirm the Python API at http://127.0.0.1:5000 is running.`
        );
      } finally {
        setDataLoading(false);
      }
    };

    loadInitialData();
  }, [authReady]); // Rerun this entire effect only when authReady changes

  // --- Handlers ---

  const handleLoadScreenLoaded = () => {
    setAppLoading(false); // This removes the full-screen logo loader
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  const addPlayerToBetSheet = (playerToAdd) => {
    if (!betSheet.find((p) => p.id === playerToAdd.id)) {
      setBetSheet((prev) => [playerToAdd, ...prev]);
    }
  };

  const removePlayerFromBetSheet = (playerId) => {
    setBetSheet((prev) => prev.filter((p) => p.id !== playerId));
  };

  // --- Render ---

  if (appLoading) {
    return <LoadingScreen onLoaded={handleLoadScreenLoaded} />;
  }

  return (
    <div className="bg-gray-100 dark:bg-gray-900 min-h-screen text-gray-900 dark:text-white transition-colors">
      <Header
        page={page}
        setPage={setPage}
        darkMode={darkMode}
        toggleDarkMode={toggleDarkMode}
        betCount={betSheet.length}
      />

      <main className="container mx-auto p-4">
        {error && (
          <motion.div
            className="my-4 p-4 bg-red-100 dark:bg-red-900 border border-red-400 text-red-700 dark:text-red-200 rounded-lg"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h3 className="font-bold">Error</h3>
            <p>{error}</p>
          </motion.div>
        )}

        {page === "home" ? (
          <HomePage
            players={allPlayers}
            setPlayers={setAllPlayers}
            addPlayerToBetSheet={addPlayerToBetSheet}
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
            hitRateFilter={hitRateFilter}
            setHitRateFilter={setHitRateFilter}
            loading={dataLoading}
            loadingProgress={loadingProgress}
            selectedGameId={selectedGameId}
            setSelectedGameId={setSelectedGameId}
            gamesList={gamesList}
            currentGameIndex={currentGameIndex}
            onLoadNextGame={loadNextGame}
          />
        ) : (
          <BetSheetPage
            betSheet={betSheet}
            removePlayerFromBetSheet={removePlayerFromBetSheet}
          />
        )}
      </main>
    </div>
  );
}
