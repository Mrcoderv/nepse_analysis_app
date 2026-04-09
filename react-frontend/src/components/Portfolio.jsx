import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getPortfolio, addStockToPortfolio, updatePortfolio, deletePortfolio } from '../services/api';
import { Briefcase, Plus, TrendingUp, TrendingDown, Wallet, PieChart, Loader2, Trash2, Edit2, Check, X, ExternalLink } from 'lucide-react';

const Portfolio = () => {
  const [holdings, setHoldings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({ symbol: '', quantity: '', buyPrice: '' });
  const [submitting, setSubmitting] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({ quantity: '', buyPrice: '' });

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
    const interval = setInterval(fetchPortfolio, 30000); // Poll every 30 seconds
    return () => clearInterval(interval);
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

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to remove this stock from your portfolio?')) return;
    try {
      await deletePortfolio(id);
      await fetchPortfolio();
    } catch (err) {
      console.error('Failed to delete stock:', err);
    }
  };

  const startEditing = (h) => {
    setEditingId(h.id);
    setEditForm({ quantity: h.quantity, buyPrice: h.buyPrice });
  };

  const handleUpdate = async (id) => {
    try {
      await updatePortfolio(id, {
        quantity: Number(editForm.quantity),
        buyPrice: Number(editForm.buyPrice)
      });
      setEditingId(null);
      await fetchPortfolio();
    } catch (err) {
      console.error('Failed to update stock:', err);
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
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-center">Signal</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">P/L %</th>
                    <th className="px-6 py-4 text-[10px] font-black text-neutral-500 uppercase tracking-tighter text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/50 text-sm">
                  {loading ? (
                    [1, 2, 3].map(i => (
                      <tr key={i} className="animate-pulse">
                        <td className="px-6 py-5"><div className="h-4 w-12 bg-neutral-800 rounded"></div></td>
                        <td className="px-6 py-5"><div className="h-4 w-8 bg-neutral-800 rounded ml-auto"></div></td>
                        <td colSpan="5"></td>
                      </tr>
                    ))
                  ) : holdings.length > 0 ? (
                    holdings.map((h) => (
                      <tr key={h.id} className="hover:bg-neutral-800/30 transition-colors group">
                        <td className="px-6 py-5">
                          <Link 
                            to={`/stock/${h.symbol}`}
                            className="flex items-center space-x-2 font-black text-blue-400 hover:text-blue-300 transition-colors"
                          >
                            <span>{h.symbol}</span>
                            <ExternalLink size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                          </Link>
                        </td>
                        <td className="px-6 py-5 text-right font-medium">
                          {editingId === h.id ? (
                            <input
                              type="number"
                              className="w-20 bg-neutral-950 border border-neutral-700 rounded px-2 py-1 text-right focus:outline-none focus:border-blue-500"
                              value={editForm.quantity}
                              onChange={e => setEditForm({...editForm, quantity: e.target.value})}
                            />
                          ) : h.quantity}
                        </td>
                        <td className="px-6 py-5 text-right text-neutral-400 font-mono">
                          {editingId === h.id ? (
                            <input
                              type="number"
                              step="0.01"
                              className="w-24 bg-neutral-950 border border-neutral-700 rounded px-2 py-1 text-right focus:outline-none focus:border-blue-500"
                              value={editForm.buyPrice}
                              onChange={e => setEditForm({...editForm, buyPrice: e.target.value})}
                            />
                          ) : `Rs. ${h.buyPrice.toFixed(2)}`}
                        </td>
                        <td className="px-6 py-5 text-right font-bold text-neutral-100">Rs. {h.currentPrice.toFixed(2)}</td>
                        <td className="px-6 py-5 text-center">
                          <span className={`px-2 py-1 rounded-full text-[10px] font-black uppercase tracking-tight ${
                            h.recommendation?.includes('buy') ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                            h.recommendation?.includes('sell') ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                            'bg-neutral-800 text-neutral-400 border border-neutral-700/50'
                          }`}>
                            {h.signal || 'HOLD'}
                          </span>
                        </td>
                        <td className={`px-6 py-5 text-right font-black ${h.percentReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {h.percentReturn >= 0 ? '+' : ''}{h.percentReturn}%
                        </td>
                        <td className="px-6 py-5 text-right">
                          <div className="flex items-center justify-end space-x-2">
                            {editingId === h.id ? (
                              <>
                                <button onClick={() => handleUpdate(h.id)} className="p-1.5 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 rounded-lg transition-all">
                                  <Check size={16} />
                                </button>
                                <button onClick={() => setEditingId(null)} className="p-1.5 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-lg transition-all">
                                  <X size={16} />
                                </button>
                              </>
                            ) : (
                              <>
                                <button onClick={() => startEditing(h)} className="p-1.5 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20 rounded-lg opacity-0 group-hover:opacity-100 transition-all">
                                  <Edit2 size={16} />
                                </button>
                                <button onClick={() => handleDelete(h.id)} className="p-1.5 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-lg opacity-0 group-hover:opacity-100 transition-all">
                                  <Trash2 size={16} />
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="6" className="px-6 py-20 text-center text-neutral-500 italic">No stocks in portfolio yet</td>
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
