"use client";

import { useState } from "react";
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CATEGORIES = [
  "Art", "Comics", "Crafts", "Dance", "Design", "Fashion",
  "Film & Video", "Food", "Games", "Journalism", "Music",
  "Photography", "Publishing", "Technology", "Theater",
];
const COUNTRIES = [
  { code: "US", name: "United States" },
  { code: "GB", name: "United Kingdom" },
  { code: "CA", name: "Canada" },
  { code: "AU", name: "Australia" },
  { code: "NZ", name: "New Zealand" },
];

type Features = {
  blurb_length: number;
  sentiment_score: number;
  readability_score: number;
  name_blurb_similarity: number;
};

type ScoreResult = {
  probability: number;
  features: Features;
  strengths: string[];
  recommendations: string[];
  similar_projects: { name: string; blurb: string }[];
};

type RewriteResult = {
  original_blurb: string;
  rewritten_blurb: string;
  original_probability: number;
  rewritten_probability: number;
  lift: number;
  original_features: Features;
  rewritten_features: Features;
};

type LeaderboardEntry = {
  project_name: string;
  blurb: string;
  category: string;
  country: string;
  goal_usd: number;
  original_score: number;
  rewritten_blurb: string;
  rewritten_score: number;
  submitted_at: string;
};

function SuccessMeter({ probability }: { probability: number }) {
  const pct = Math.round(probability * 100);
  const color =
    pct >= 60 ? "#05ce78" : pct >= 40 ? "#f39c12" : "#e74c3c";
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-gray-600">Predicted Success</span>
        <span className="text-2xl font-bold" style={{ color }}>{pct}%</span>
      </div>
      <div className="success-meter">
        <div
          className="success-meter-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function FeatureBar({ label, value, max, unit }: {
  label: string; value: number; max: number; unit?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const displayValue = Number.isInteger(value) ? value.toString() : value.toFixed(2);

  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm mb-1.5">
        <span className="font-medium text-gray-700">{label}</span>
        <span className="font-bold text-[#282828]">
          {displayValue}{unit || ""}
        </span>
      </div>
      <div className="relative h-2.5 rounded-full bg-gray-100">
        <div
          className="h-full rounded-full bg-[#05ce78] transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>0</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function RadarChartComponent({
  original, rewritten,
}: { original: Features; rewritten?: Features }) {
  const normalize = (val: number, min: number, max: number) =>
    Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));

  const data = [
    {
      feature: "Sentiment",
      Original: normalize(original.sentiment_score, -1, 1),
      ...(rewritten && { Rewritten: normalize(rewritten.sentiment_score, -1, 1) }),
    },
    {
      feature: "Readability",
      Original: normalize(original.readability_score, 0, 100),
      ...(rewritten && { Rewritten: normalize(rewritten.readability_score, 0, 100) }),
    },
    {
      feature: "Length",
      Original: normalize(original.blurb_length, 0, 40),
      ...(rewritten && { Rewritten: normalize(rewritten.blurb_length, 0, 40) }),
    },
    {
      feature: "Coherence",
      Original: normalize(original.name_blurb_similarity, 0, 1),
      ...(rewritten && { Rewritten: normalize(rewritten.name_blurb_similarity, 0, 1) }),
    },
  ];

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data}>
        <PolarGrid stroke="#e0e0e0" />
        <PolarAngleAxis dataKey="feature" tick={{ fontSize: 12 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
        <Radar name="Original" dataKey="Original" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.2} />
        {rewritten && (
          <Radar name="Rewritten" dataKey="Rewritten" stroke="#05ce78" fill="#05ce78" fillOpacity={0.2} />
        )}
        <Legend />
      </RadarChart>
    </ResponsiveContainer>
  );
}

export default function Home() {
  const [form, setForm] = useState({
    project_name: "",
    blurb: "",
    category: "Technology",
    country: "US",
    goal_usd: 10000,
    duration: 30,
  });

  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [rewriteResult, setRewriteResult] = useState<RewriteResult | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [rewriting, setRewriting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [activeTab, setActiveTab] = useState<"analyze" | "leaderboard">("analyze");

  const handleScore = async () => {
    setLoading(true);
    setScoreResult(null);
    setRewriteResult(null);
    setSubmitted(false);
    try {
      const res = await fetch(`${API_URL}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setScoreResult(data);
    } catch {
      alert("Error connecting to backend. Make sure the API is running.");
    }
    setLoading(false);
  };

  const handleRewrite = async () => {
    setRewriting(true);
    try {
      const res = await fetch(`${API_URL}/rewrite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setRewriteResult(data);
    } catch {
      alert("Error rewriting blurb.");
    }
    setRewriting(false);
  };

  const handleSubmit = async () => {
    if (!rewriteResult) return;
    try {
      await fetch(`${API_URL}/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          original_score: rewriteResult.original_probability,
          rewritten_blurb: rewriteResult.rewritten_blurb,
          rewritten_score: rewriteResult.rewritten_probability,
        }),
      });
      setSubmitted(true);
      fetchLeaderboard();
    } catch {
      alert("Error submitting.");
    }
  };

  const fetchLeaderboard = async () => {
    try {
      const res = await fetch(`${API_URL}/leaderboard`);
      const data = await res.json();
      setLeaderboard(data.leaderboard);
    } catch { /* ignore */ }
  };

  return (
    <div className="min-h-screen bg-[#f7f7f7]">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#05ce78] flex items-center justify-center">
              <span className="text-white font-bold text-sm">K</span>
            </div>
            <h1 className="text-xl font-bold text-[#282828]">Blurb Optimizer</h1>
          </div>
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`px-4 py-2 rounded-md text-sm font-medium transition ${
                activeTab === "analyze"
                  ? "bg-white shadow text-[#282828]"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Analyze
            </button>
            <button
              onClick={() => { setActiveTab("leaderboard"); fetchLeaderboard(); }}
              className={`px-4 py-2 rounded-md text-sm font-medium transition ${
                activeTab === "leaderboard"
                  ? "bg-white shadow text-[#282828]"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Leaderboard
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {activeTab === "analyze" ? (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            {/* Left: Form */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 className="text-lg font-bold mb-6 text-[#282828]">Your Project</h2>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Project Name
                    </label>
                    <input
                      type="text"
                      value={form.project_name}
                      onChange={(e) => setForm({ ...form, project_name: e.target.value })}
                      placeholder="My Amazing Project"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#05ce78] focus:border-transparent outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Blurb
                    </label>
                    <textarea
                      value={form.blurb}
                      onChange={(e) => setForm({ ...form, blurb: e.target.value })}
                      placeholder="Describe your project in 1-2 sentences..."
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#05ce78] focus:border-transparent outline-none resize-none"
                    />
                    <div className="text-xs text-gray-400 mt-1">
                      {form.blurb.split(/\s+/).filter(Boolean).length} words
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Category
                      </label>
                      <select
                        value={form.category}
                        onChange={(e) => setForm({ ...form, category: e.target.value })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg outline-none"
                      >
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Country
                      </label>
                      <select
                        value={form.country}
                        onChange={(e) => setForm({ ...form, country: e.target.value })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg outline-none"
                      >
                        {COUNTRIES.map((c) => (
                          <option key={c.code} value={c.code}>{c.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Goal (USD)
                      </label>
                      <input
                        type="number"
                        value={form.goal_usd}
                        onChange={(e) => setForm({ ...form, goal_usd: Number(e.target.value) })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Duration (days)
                      </label>
                      <input
                        type="number"
                        value={form.duration}
                        onChange={(e) => setForm({ ...form, duration: Number(e.target.value) })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg outline-none"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleScore}
                    disabled={loading || !form.project_name || !form.blurb}
                    className="w-full py-3 bg-[#05ce78] hover:bg-[#028858] disabled:bg-gray-300 text-white font-bold rounded-lg transition cursor-pointer disabled:cursor-not-allowed"
                  >
                    {loading ? "Analyzing..." : "Analyze My Blurb"}
                  </button>
                </div>
              </div>
            </div>

            {/* Right: Results */}
            <div className="lg:col-span-3 space-y-6">
              {!scoreResult && !loading && (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center text-gray-400">
                  <div className="text-5xl mb-4">&#9998;</div>
                  <p className="text-lg">Enter your project details and click <strong>Analyze</strong></p>
                  <p className="text-sm mt-2">We will predict your success probability and help you improve your blurb</p>
                </div>
              )}

              {scoreResult && (
                <>
                  {/* Success Meter */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <SuccessMeter probability={scoreResult.probability} />
                  </div>

                  {/* Strengths & Recommendations */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="font-bold text-[#282828] mb-4">Analysis</h3>
                    {scoreResult.strengths.length > 0 && (
                      <div className="mb-4">
                        <p className="text-sm font-medium text-green-700 mb-2">Strengths</p>
                        {scoreResult.strengths.map((s, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm text-gray-700 mb-1">
                            <span className="text-green-500 mt-0.5">{"\u2713"}</span> {s}
                          </div>
                        ))}
                      </div>
                    )}
                    {scoreResult.recommendations.length > 0 && (
                      <div>
                        <p className="text-sm font-medium text-orange-600 mb-2">Recommendations</p>
                        {scoreResult.recommendations.map((r, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm text-gray-700 mb-1">
                            <span className="text-orange-500 mt-0.5">{"\u26A0"}</span> {r}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Feature Breakdown */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="font-bold text-[#282828] mb-4">
                      Feature Breakdown
                      {rewriteResult && <span className="text-sm font-normal text-gray-400 ml-2">(showing rewritten blurb)</span>}
                    </h3>
                    {(() => {
                      const feats = rewriteResult ? rewriteResult.rewritten_features : scoreResult.features;
                      return (
                        <>
                          <FeatureBar label="Sentiment Score" value={feats.sentiment_score} max={1} />
                          <FeatureBar label="Readability (Flesch)" value={feats.readability_score} max={100} />
                          <FeatureBar label="Blurb Length" value={feats.blurb_length} max={40} unit=" words" />
                          <FeatureBar label="Name-Blurb Coherence" value={feats.name_blurb_similarity} max={1} />
                        </>
                      );
                    })()}
                  </div>

                  {/* Radar Chart */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="font-bold text-[#282828] mb-4">
                      {rewriteResult ? "Before vs. After" : "Feature Overview"}
                    </h3>
                    <RadarChartComponent
                      original={scoreResult.features}
                      rewritten={rewriteResult?.rewritten_features}
                    />
                  </div>

                  {/* Rewrite Button */}
                  {!rewriteResult && (
                    <button
                      onClick={handleRewrite}
                      disabled={rewriting}
                      className="w-full py-3 bg-[#282828] hover:bg-[#3d3d3d] disabled:bg-gray-400 text-white font-bold rounded-lg transition cursor-pointer disabled:cursor-not-allowed"
                    >
                      {rewriting ? "Rewriting with AI..." : "Improve My Blurb with AI"}
                    </button>
                  )}

                  {/* Rewrite Result */}
                  {rewriteResult && (
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                      <h3 className="font-bold text-[#282828] mb-4">AI-Improved Blurb</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="p-4 bg-red-50 rounded-lg border border-red-100">
                          <p className="text-xs font-medium text-red-600 mb-2">ORIGINAL</p>
                          <p className="text-sm text-gray-700">{rewriteResult.original_blurb}</p>
                          <p className="text-lg font-bold text-red-600 mt-2">
                            {Math.round(rewriteResult.original_probability * 100)}%
                          </p>
                        </div>
                        <div className="p-4 bg-green-50 rounded-lg border border-green-100">
                          <p className="text-xs font-medium text-green-600 mb-2">IMPROVED</p>
                          <p className="text-sm text-gray-700">{rewriteResult.rewritten_blurb}</p>
                          <p className="text-lg font-bold text-green-600 mt-2">
                            {Math.round(rewriteResult.rewritten_probability * 100)}%
                            <span className="text-sm ml-2">
                              ({rewriteResult.lift >= 0 ? "+" : ""}
                              {Math.round(rewriteResult.lift * 100)}pp)
                            </span>
                          </p>
                        </div>
                      </div>
                      {!submitted && (
                        <button
                          onClick={handleSubmit}
                          className="w-full py-3 bg-[#05ce78] hover:bg-[#028858] text-white font-bold rounded-lg transition cursor-pointer"
                        >
                          Submit to Leaderboard
                        </button>
                      )}
                      {submitted && (
                        <div className="text-center py-3 bg-green-50 rounded-lg text-green-700 font-medium">
                          {"\u2713"} Submitted! Check the leaderboard.
                        </div>
                      )}
                    </div>
                  )}

                </>
              )}
            </div>
          </div>
        ) : (
          /* Leaderboard */
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-[#282828]">Top 10 Projects</h2>
                <button
                  onClick={fetchLeaderboard}
                  className="text-sm text-[#05ce78] hover:text-[#028858] font-medium cursor-pointer"
                >
                  Refresh
                </button>
              </div>

              {leaderboard.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <div className="text-4xl mb-3">&#127942;</div>
                  <p>No submissions yet. Be the first!</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {leaderboard.map((entry, i) => (
                    <div
                      key={i}
                      className="leaderboard-row flex items-center gap-4 p-4 bg-gray-50 rounded-lg border border-gray-100"
                      style={{ animationDelay: `${i * 0.05}s` }}
                    >
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-white shrink-0 ${
                        i === 0 ? "bg-yellow-400" : i === 1 ? "bg-gray-400" : i === 2 ? "bg-orange-400" : "bg-gray-300"
                      }`}>
                        {i + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-[#282828] truncate">{entry.project_name}</p>
                        <p className="text-xs text-gray-500 truncate">{entry.rewritten_blurb}</p>
                        <p className="text-xs text-gray-400">{entry.category} &middot; ${entry.goal_usd.toLocaleString()}</p>
                      </div>
                      <div className="text-right shrink-0">
                        <p className="text-xl font-bold text-[#05ce78]">
                          {Math.round(entry.rewritten_score * 100)}%
                        </p>
                        <p className="text-xs text-gray-400">
                          from {Math.round(entry.original_score * 100)}%
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-gray-200 bg-white py-4">
        <div className="max-w-5xl mx-auto px-4 text-center text-xs text-gray-400">
          Text Analysis for Business &middot; Imperial College London &middot; Crowdfunding Success Prediction
        </div>
      </footer>
    </div>
  );
}
