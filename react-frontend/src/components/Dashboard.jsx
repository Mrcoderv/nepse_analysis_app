import React, { useEffect, useState } from 'react';
import { getNepseIndex, getTopGainers, getTopLosers } from '../services/api';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3 } from 'lucide-react';

const Dashboard = () => {
  const [indexData, setIndexData] = useState(null);
  const [gainers, setGainers] = useState([]);
  const [losers, setLosers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [indexRes, gainersRes, losersRes] = await Promise.all([
          getNepseIndex(),
          getTopGainers(),
          getTopLosers()
        ]);
        setIndexData(indexRes.data);
        setGainers(Array.isArray(gainersRes.data) ? gainersRes.data.slice(0, 5) : []);
        setLosers(Array.isArray(losersRes.data) ? losersRes.data.slice(0, 5) : []);
      } catch (err) {
        console.error('Failed to fetch dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-32 bg-neutral-900 border border-neutral-800 rounded-2xl animate-pulse"></div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
           <div className="h-96 bg-neutral-900 border border-neutral-800 rounded-2xl animate-pulse"></div>
           <div className="h-96 bg-neutral-900 border border-neutral-800 rounded-2xl animate-pulse"></div>
        </div>
      </div>
    );
  }

  // Handle nested index data if needed (depends on @rumess/nepse-api response)
  const mainIndex = Array.isArray(indexData) ? indexData[0] : indexData;

  const stats = [
    { name: 'NEPSE Index', value: mainIndex?.close || '0.00', change: mainIndex?.change || '0.00', percent: mainIndex?.perChange ? `${mainIndex.perChange}%` : '0.00%', icon: Activity, color: 'blue' },
    { name: 'Daily Volume', value: 'N/A', change: '', percent: '', icon: BarChart3, color: 'emerald' },
    { name: 'Total Turnover', value: 'N/A', change: '', percent: '', icon: DollarSign, color: 'amber' },
    { name: 'Market Cap', value: 'N/A', change: '', percent: '', icon: TrendingUp, color: 'purple' },
  ];

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        {stats.map((stat, i) => (
          <div key={i} className="p-6 bg-neutral-900/50 border border-neutral-800 rounded-2xl hover:border-neutral-700 transition-all group">
            <div className="flex items-center justify-between">
              <div className={`p-2 rounded-xl bg-${stat.color}-500/10 text-${stat.color}-400 group-hover:scale-110 transition-transform`}>
                <stat.icon size={20} />
              </div>
              {stat.percent && (
                <span className={`text-xs font-semibold px-2 py-1 rounded-full ${String(stat.percent).startsWith('-') ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                  {stat.percent}
                </span>
              )}
            </div>
            <div className="mt-4">
              <p className="text-sm font-medium text-neutral-400">{stat.name}</p>
              <h3 className="text-2xl font-bold mt-1 text-neutral-100">{stat.value}</h3>
              {stat.change && <p className={`text-xs mt-1 ${String(stat.change).startsWith('-') ? 'text-red-400' : 'text-emerald-400'}`}>{stat.change} today</p>}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Top Gainers */}
        <div className="p-6 bg-neutral-900/50 border border-neutral-800 rounded-2xl backdrop-blur-sm">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
                <TrendingUp size={20} />
              </div>
              <h3 className="text-lg font-bold">Top Gainers</h3>
            </div>
            <button className="text-xs font-semibold text-blue-400 hover:text-blue-300">View All</button>
          </div>
          <div className="space-y-4">
            {gainers.map((stock, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-neutral-800/30 rounded-xl hover:bg-neutral-800/50 transition-colors border border-transparent hover:border-neutral-700 cursor-pointer">
                <div>
                  <p className="font-bold text-neutral-100">{stock.symbol}</p>
                  <p className="text-xs text-neutral-500">{stock.securityName || 'Company Name'}</p>
                </div>
                <div className="text-right">
                  <p className="font-bold text-neutral-100">Rs. {stock.ltp}</p>
                  <p className="text-xs font-semibold text-emerald-400">+{stock.percentageChange}%</p>
                </div>
              </div>
            ))}
            {gainers.length === 0 && <p className="text-center py-8 text-neutral-500 italic">No gainer data available</p>}
          </div>
        </div>

        {/* Top Losers */}
        <div className="p-6 bg-neutral-900/50 border border-neutral-800 rounded-2xl backdrop-blur-sm">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-500/10 text-red-400 rounded-lg">
                <TrendingDown size={20} />
              </div>
              <h3 className="text-lg font-bold">Top Losers</h3>
            </div>
            <button className="text-xs font-semibold text-blue-400 hover:text-blue-300">View All</button>
          </div>
          <div className="space-y-4">
            {losers.map((stock, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-neutral-800/30 rounded-xl hover:bg-neutral-800/50 transition-colors border border-transparent hover:border-neutral-700 cursor-pointer">
                <div>
                  <p className="font-bold text-neutral-100">{stock.symbol}</p>
                  <p className="text-xs text-neutral-500">{stock.securityName || 'Company Name'}</p>
                </div>
                <div className="text-right">
                  <p className="font-bold text-neutral-100">Rs. {stock.ltp}</p>
                  <p className="text-xs font-semibold text-red-400">{stock.percentageChange}%</p>
                </div>
              </div>
            ))}
             {losers.length === 0 && <p className="text-center py-8 text-neutral-500 italic">No loser data available</p>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
