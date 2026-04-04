import React, { useEffect, useState } from 'react';
import { getPortfolio, addStockToPortfolio } from '../services/api';
import { Briefcase, Plus, TrendingUp, TrendingDown, Wallet, PieChart, Loader2 } from 'lucide-react';

const Portfolio = () => {
  const [holdings, setHoldings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({ symbol: '', quantity: '', buyPrice: '' });
  const [submitting, setSubmitting] = useState(false);

  const fetchPortfolio = async () => {
    try {
      const res = await getPortfolio();
      setHoldings(res.data);
    } catch (err) {
      console.error('Failed to fetch portfolio:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.symbol || !form.quantity || !form.buyPrice) return;
    setSubmitting(true);
    try {
      await addStockToPortfolio({
        symbol: form.symbol.toUpperCase(),
        quantity: Number(form.quantity),
        buyPrice: Number(form.buyPrice)
      });
      setForm({ symbol: '', quantity: '', buyPrice: '' });
      await fetchPortfolio();
    } catch (err) {
      console.error('Failed to add stock:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const totalInvestment = holdings.reduce((acc, h) => acc + h.totalInvestment, 0);
  const currentValue = holdings.reduce((acc, h) => acc + h.currentValue, 0);
  const totalPL = currentValue - totalInvestment;
  const totalPLPercent = totalInvestment > 0 ? (totalPL / totalInvestment) * 100 : 0;

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-20">
      {/* Portfolio Header Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-3xl relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 opacity-5 group-hover:scale-110 transition-transform duration-700">
             <Wallet size={120} />
          </div>
          <p className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-1">Total Investment</p>
          <h3 className="text-2xl font-black text-neutral-100">Rs. {totalInvestment.toLocaleString()}</h3>
        </div>

        <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-3xl relative overflow-hidden group">
           <div className="absolute -right-4 -top-4 opacity-5 group-hover:scale-110 transition-transform duration-700">
             <PieChart size={120} />
          </div>
          <p className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-1">Current Value</p>
          <h3 className="text-2xl font-black text-neutral-100">Rs. {currentValue.toLocaleString()}</h3>
        </div>

        <div className={`p-6 border rounded-3xl relative overflow-hidden group ${totalPL >= 0 ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'}`}>
          <div className="absolute -right-4 -top-4 opacity-5 group-hover:scale-110 transition-transform duration-700">
             {totalPL >= 0 ? <TrendingUp size={120} /> : <TrendingDown size={120} />}
          </div>
          <p className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-1">Overall Profit/Loss</p>
          <div className="flex items-baseline space-x-2">
            <h3 className={`text-2xl font-black ${totalPL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {totalPL >= 0 ? '+' : ''}{totalPL.toLocaleString()}
            </h3>
            <span className={`text-sm font-bold ${totalPL >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
              ({totalPLPercent.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Holdings Table */}
        <div className="xl:col-span-2 bg-neutral-900/50 border border-neutral-800 rounded-3xl overflow-hidden backdrop-blur-sm">
          <div className="p-6 border-b border-neutral-800 flex items-center justify-between">
            <h3 className="text-lg font-bold flex items-center">
              <Briefcase size={20} className="mr-3 text-blue-500" /> Current Holdings
            </h3>
          </div>
          <div className="overflow-x-auto">
             <table className="w-full text-left">
                <thead className="bg-neutral-950/50 border-b border-neutral-800">
                  <tr>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter">Stock</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">Qty</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">Avg. Price</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">Current</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">P/L %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/50 text-sm">
                  {loading ? (
                    [1, 2, 3].map(i => (
                      <tr key={i} className="animate-pulse">
                        <td className="px-6 py-5"><div className="h-4 w-12 bg-neutral-800 rounded"></div></td>
                        <td className="px-6 py-5"><div className="h-4 w-8 bg-neutral-800 rounded ml-auto"></div></td>
                        <td colSpan="3"></td>
                      </tr>
                    ))
                  ) : holdings.length > 0 ? (
                    holdings.map((h, i) => (
                      <tr key={i} className="hover:bg-neutral-800/30 transition-colors">
                        <td className="px-6 py-5 font-black text-blue-400">{h.symbol}</td>
                        <td className="px-6 py-5 text-right font-medium">{h.quantity}</td>
                        <td className="px-6 py-5 text-right text-neutral-400 font-mono">Rs. {h.buyPrice.toFixed(2)}</td>
                        <td className="px-6 py-5 text-right font-bold text-neutral-100">Rs. {h.currentPrice.toFixed(2)}</td>
                        <td className={`px-6 py-5 text-right font-black ${h.percentReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {h.percentReturn >= 0 ? '+' : ''}{h.percentReturn}%
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="5" className="px-6 py-20 text-center text-neutral-500 italic">No stocks in portfolio yet</td>
                    </tr>
                  )}
                </tbody>
             </table>
          </div>
        </div>

        {/* Add Stock Form */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-8 h-fit shadow-2xl">
          <h3 className="text-xl font-bold mb-6 flex items-center">
            <Plus size={20} className="mr-3 text-blue-400" /> New Transaction
          </h3>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-[10px] font-black text-neutral-500 uppercase tracking-widest mb-2">Stock Symbol</label>
              <input
                type="text"
                placeholder="e.g. NICA"
                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl py-3 px-4 focus:outline-none focus:ring-2 focus:ring-blue-600/50 transition-all font-bold"
                value={form.symbol}
                onChange={e => setForm({...form, symbol: e.target.value})}
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-black text-neutral-500 uppercase tracking-widest mb-2">Quantity</label>
                <input
                  type="number"
                  placeholder="0"
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-xl py-3 px-4 focus:outline-none focus:ring-2 focus:ring-blue-600/50 transition-all"
                  value={form.quantity}
                  onChange={e => setForm({...form, quantity: e.target.value})}
                  required
                />
              </div>
              <div>
                <label className="block text-[10px] font-black text-neutral-500 uppercase tracking-widest mb-2">Buy Price</label>
                <input
                  type="number"
                  step="0.01"
                  placeholder="0.00"
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-xl py-3 px-4 focus:outline-none focus:ring-2 focus:ring-blue-600/50 transition-all"
                  value={form.buyPrice}
                  onChange={e => setForm({...form, buyPrice: e.target.value})}
                  required
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={submitting}
              className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-black rounded-2xl transition-all shadow-lg shadow-blue-600/20 flex items-center justify-center"
            >
              {submitting ? <Loader2 className="animate-spin mr-2" size={20} /> : <Plus size={20} className="mr-2" />}
              ADD TO PORTFOLIO
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
