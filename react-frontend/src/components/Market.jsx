import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCompanyList } from '../services/api';
import { Search, ChevronRight, Building2, Hash } from 'lucide-react';

const Market = () => {
  const [companies, setCompanies] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCompanies = async () => {
      try {
        const res = await getCompanyList();
        setCompanies(res.data);
      } catch (err) {
        console.error('Failed to fetch company list:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchCompanies();
  }, []);

  const filtered = companies.filter(c => 
    c.symbol?.toLowerCase().includes(search.toLowerCase()) || 
    c.name?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-neutral-100">Market Explorer</h2>
        <div className="relative w-full md:w-96">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500" size={18} />
          <input
            type="text"
            placeholder="Search symbol or company name..."
            className="w-full bg-neutral-900 border border-neutral-800 rounded-xl py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600/50 transition-all"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl overflow-hidden backdrop-blur-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-neutral-950/50 border-b border-neutral-800">
              <tr>
                <th className="px-6 py-4 text-xs font-bold text-neutral-500 uppercase tracking-wider">Symbol</th>
                <th className="px-6 py-4 text-xs font-bold text-neutral-500 uppercase tracking-wider">Company Name</th>
                <th className="px-6 py-4 text-xs font-bold text-neutral-500 uppercase tracking-wider text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/50">
              {loading ? (
                [1, 2, 3, 4, 5].map(i => (
                  <tr key={i} className="animate-pulse">
                    <td className="px-6 py-4"><div className="h-4 w-16 bg-neutral-800 rounded"></div></td>
                    <td className="px-6 py-4"><div className="h-4 w-64 bg-neutral-800 rounded"></div></td>
                    <td className="px-6 py-4"><div className="h-4 w-8 bg-neutral-800 rounded ml-auto"></div></td>
                  </tr>
                ))
              ) : filtered.length > 0 ? (
                filtered.map((company) => (
                  <tr 
                    key={company.symbol} 
                    className="hover:bg-neutral-800/50 transition-colors group cursor-pointer"
                    onClick={() => navigate(`/stock/${company.symbol}`)}
                  >
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-2 py-1 bg-blue-500/10 text-blue-400 text-xs font-bold rounded-lg border border-blue-500/20">
                        {company.symbol}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center">
                        <div className="p-2 bg-neutral-800 rounded-lg mr-3 text-neutral-500 group-hover:text-neutral-300 transition-colors">
                           <Building2 size={16} />
                        </div>
                        <span className="text-sm font-medium text-neutral-300 group-hover:text-neutral-100 transition-colors">
                          {company.securityName}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button className="p-2 text-neutral-500 hover:text-blue-400 transition-colors">
                        <ChevronRight size={18} />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" className="px-6 py-12 text-center text-neutral-500 italic">
                    No companies found matching "{search}"
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Market;
