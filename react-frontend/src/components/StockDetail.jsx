import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getStockHistory, getStockAnalysis, getAIAnalysis, getCompanyDetails } from '../services/api';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';
import { Brain, Zap, ShieldCheck, AlertCircle, Info, TrendingUp, TrendingDown, Target } from 'lucide-react';

const StockDetail = () => {
  const { symbol } = useParams();
  const [history, setHistory] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [aiResult, setAiResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [histRes, analRes] = await Promise.all([
          getStockHistory(symbol),
          getStockAnalysis(symbol)
        ]);
        
        // Transform history for Recharts
        const data = (histRes.data.content || []).map(item => ({
          date: item.businessDate,
          price: item.closePrice,
          volume: item.totalTradedQuantity
        })).sort((a, b) => new Date(a.date) - new Date(b.date));
        
        setHistory(data);
        setAnalysis(analRes.data);
      } catch (err) {
        console.error('Failed to fetch stock details:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [symbol]);

  const handleAIAnalysis = async () => {
    setAiLoading(true);
    try {
      const res = await getAIAnalysis(symbol);
      setAiResult(res.data);
    } catch (err) {
      console.error('AI Analysis failed:', err);
    } finally {
      setAiLoading(false);
    }
  };

  if (loading) {
    return <div className="h-screen flex items-center justify-center text-neutral-500 animate-pulse">Analyzing Market Data for {symbol}...</div>;
  }

  const indicators = analysis?.indicators || {};

  return (
    <div className="space-y-8 pb-12 animate-in fade-in slide-in-from-right-4 duration-500">
      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-neutral-800 pb-6 gap-4">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <span className="px-3 py-1 bg-blue-600 text-white text-sm font-bold rounded-lg shadow-lg shadow-blue-600/20">{symbol}</span>
            <h2 className="text-3xl font-bold text-neutral-100 italic">{analysis?.lastPrice && `Rs. ${analysis.lastPrice}`}</h2>
            <div className={`ml-4 px-4 py-1.5 rounded-xl border-2 font-black tracking-tighter text-sm ${
              analysis?.recommendation === 'strong_buy' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
              analysis?.recommendation === 'buy' ? 'bg-emerald-500/5 text-emerald-500 border-emerald-500/20' :
              analysis?.recommendation === 'strong_sell' ? 'bg-red-500/10 text-red-400 border-red-500/30' :
              analysis?.recommendation === 'sell' ? 'bg-red-500/5 text-red-500 border-red-500/20' :
              'bg-amber-500/10 text-amber-400 border-amber-500/30'
            }`}>
              {analysis?.signal_level || 'HOLD'}
            </div>
          </div>
          <div className="flex items-center space-x-4 text-sm mt-3">
            <div className={`flex items-center ${analysis?.trend === 'BULLISH' ? 'text-emerald-400' : analysis?.trend === 'BEARISH' ? 'text-red-400' : 'text-neutral-400'}`}>
              {analysis?.trend === 'BULLISH' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span className="ml-1 font-bold">{analysis?.trend} TREND</span>
            </div>
            <span className="text-neutral-600">|</span>
            <span className="text-neutral-400 font-medium tracking-tight">EFFICIENCY: <span className="text-white font-black">{analysis?.efficiency_score?.toFixed(0)}%</span></span>
          </div>
        </div>
        
        <button 
          onClick={handleAIAnalysis}
          disabled={aiLoading}
          className={`flex items-center px-6 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold rounded-2xl transition-all shadow-xl shadow-violet-600/20 group ${aiLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Brain className={`mr-2 group-hover:scale-110 transition-transform ${aiLoading ? 'animate-pulse' : ''}`} size={20} />
          {aiLoading ? 'Gemini Thinking...' : 'Ask AI Insight'}
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Main Chart */}
        <div className="xl:col-span-2 space-y-6">
          <div className="p-6 bg-neutral-900/50 border border-neutral-800 rounded-3xl overflow-hidden shadow-2xl relative">
            <h3 className="text-sm font-semibold text-neutral-500 mb-6 flex items-center">
              <Zap size={14} className="mr-2 text-amber-500" /> PRICE ACTION HISTORY
            </h3>
            
            {/* Efficiency Meter (Floating) */}
            <div className="absolute top-6 right-6 flex items-center space-x-2">
               <div className="text-right">
                  <p className="text-[10px] font-bold text-neutral-500 uppercase">Analysis Score</p>
                  <p className="text-lg font-black text-neutral-100">{analysis?.efficiency_score?.toFixed(0)}</p>
               </div>
               <div className="w-12 h-12 rounded-full border-4 border-neutral-800 relative flex items-center justify-center">
                  <div 
                    className={`absolute inset-0 rounded-full border-4 ${
                      analysis?.efficiency_score > 65 ? 'border-emerald-500' : 
                      analysis?.efficiency_score < 35 ? 'border-red-500' : 'border-amber-500'
                    }`}
                    style={{ 
                      clipPath: `polygon(50% 50%, -50% -50%, ${analysis?.efficiency_score}% -50%, ${analysis?.efficiency_score}% 150%, -50% 150%)`,
                      transform: 'rotate(-90deg)'
                    }}
                  ></div>
                  <Target size={16} className="text-neutral-500" />
               </div>
            </div>

            <div className="h-[400px] w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="date" stroke="#525252" fontSize={10} tickFormatter={(val) => val.split('T')[0]} />
                  <YAxis stroke="#525252" fontSize={10} domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#171717', border: '1px solid #404040', borderRadius: '12px' }}
                    labelStyle={{ color: '#a3a3a3', marginBottom: '4px' }}
                  />
                  <Area type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorPrice)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Result Area */}
          {aiResult && (
            <div className="p-8 bg-neutral-900 border border-violet-500/30 rounded-3xl shadow-2xl relative overflow-hidden animate-in zoom-in-95 duration-500">
               <div className="absolute top-0 right-0 p-4 opacity-5">
                  <Brain size={120} />
               </div>
               <div className="relative z-10">
                 <div className="flex items-center space-x-3 mb-6">
                    <div className="p-2 bg-violet-500/10 text-violet-400 rounded-xl border border-violet-500/20">
                       <Target size={24} />
                    </div>
                    <h3 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">Gemini AI Recommendation</h3>
                 </div>
                 
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                    <div>
                        <p className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-2">RECOMMENDATION</p>
                        <span className={`text-2xl font-black px-4 py-2 rounded-xl border-2 ${
                            aiResult.recommendation?.toUpperCase() === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
                            aiResult.recommendation?.toUpperCase() === 'SELL' ? 'bg-red-500/10 text-red-400 border-red-500/30' :
                            'bg-amber-500/10 text-amber-400 border-amber-500/30'
                        }`}>
                            {aiResult.recommendation}
                        </span>
                    </div>
                    <div>
                        <p className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-2">RISK LEVEL</p>
                        <span className="text-lg font-bold text-neutral-100">{aiResult.risk}</span>
                    </div>
                 </div>

                 <div className="space-y-4">
                    <div className="p-4 bg-neutral-950/50 rounded-2xl border border-neutral-800">
                        <p className="text-sm text-neutral-400 leading-relaxed"><span className="text-neutral-100 font-bold block mb-2 uppercase text-xs tracking-tighter">Analysis Brief</span> {aiResult.reason}</p>
                    </div>
                    <div className="p-4 bg-neutral-950/50 rounded-2xl border border-neutral-800">
                        <p className="text-sm text-neutral-400 leading-relaxed"><span className="text-neutral-100 font-bold block mb-2 uppercase text-xs tracking-tighter">Outlook</span> {aiResult.outlook}</p>
                    </div>
                 </div>
               </div>
            </div>
          )}
        </div>

        {/* Sidebar Analysis */}
        <div className="space-y-6">
          {/* Signal indicators board */}
          <div className="p-6 bg-neutral-900/50 border border-neutral-800 rounded-3xl">
            <div className="flex items-center justify-between mb-6">
               <h3 className="text-sm font-semibold text-neutral-500 flex items-center">
                 <ShieldCheck size={14} className="mr-2 text-emerald-500" /> TECHNICAL SIGNALS
               </h3>
               <span className="text-[10px] font-bold text-neutral-600 px-2 py-0.5 border border-neutral-800 rounded-md">LIVE</span>
            </div>
            <div className="space-y-3">
              {analysis?.signals?.map((sig, i) => (
                <div key={i} className={`p-4 rounded-2xl text-[11px] font-black border flex items-center transform hover:scale-[1.02] transition-transform ${
                    sig.includes('Bullish') || sig.includes('Lower') || sig.includes('Confirmation') || sig.includes('Above') || sig.includes('Oversold') ? 'bg-emerald-500/5 text-emerald-400 border-emerald-500/10' : 
                    sig.includes('Bearish') || sig.includes('Upper') || sig.includes('Death') || sig.includes('Below') || sig.includes('Overbought') ? 'bg-red-500/5 text-red-400 border-red-500/10' :
                    'bg-neutral-800/40 text-neutral-400 border-neutral-700'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full mr-3 ${
                    sig.includes('Bullish') || sig.includes('Lower') || sig.includes('Confirmation') || sig.includes('Above') || sig.includes('Oversold') ? 'bg-emerald-500' : 
                    sig.includes('Bearish') || sig.includes('Upper') || sig.includes('Death') || sig.includes('Below') || sig.includes('Overbought') ? 'bg-red-500' : 'bg-neutral-600'
                  }`}></div>
                  <span className="uppercase tracking-tight">{sig}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Key data points grid */}
          <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-3xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-violet-500 to-emerald-500"></div>
            <h3 className="text-sm font-semibold text-neutral-500 mb-6 flex items-center">
              <Info size={14} className="mr-2 text-blue-500" /> SENSOR METRICS
            </h3>
            <div className="grid grid-cols-1 gap-4">
              <div className="flex justify-between items-center p-3 border-b border-neutral-800 hover:bg-neutral-800/20 rounded-xl transition-colors">
                <span className="text-xs text-neutral-500 font-medium tracking-tight">RSI (14)</span>
                <span className={`text-sm font-black ${indicators.currentRSI > 70 ? 'text-red-400' : indicators.currentRSI < 30 ? 'text-emerald-400' : 'text-blue-400'}`}>
                  {indicators.currentRSI?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 border-b border-neutral-800 hover:bg-neutral-800/20 rounded-xl transition-colors">
                <span className="text-xs text-neutral-500 font-medium tracking-tight">MACD Signal</span>
                <span className="text-sm font-black text-neutral-100">{indicators.currentMACD?.signal?.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between items-center p-3 border-b border-neutral-800 hover:bg-neutral-800/20 rounded-xl transition-colors">
                <span className="text-xs text-neutral-500 font-medium tracking-tight">BB Compression</span>
                <span className="text-sm font-black text-neutral-300">
                   {indicators.currentBB ? ((indicators.currentBB.upper - indicators.currentBB.lower) / indicators.currentBB.middle * 100).toFixed(1) + '%' : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 hover:bg-neutral-800/20 rounded-xl transition-colors">
                <span className="text-xs text-neutral-500 font-medium tracking-tight">Volume Ratio</span>
                <span className={`text-sm font-black ${indicators.lastVolume > indicators.currentVolSMA * 1.5 ? 'text-emerald-400 animate-pulse' : 'text-neutral-400'}`}>
                   {(indicators.lastVolume / (indicators.currentVolSMA || 1)).toFixed(2)}x
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockDetail;
