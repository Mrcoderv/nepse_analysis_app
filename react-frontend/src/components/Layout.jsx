import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, ChartLine, Briefcase, Info, Menu, X } from 'lucide-react';

import Footer from './Footer';

const Layout = ({ children }) => {
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(true);
  const [isMarketOpen, setIsMarketOpen] = React.useState(true);
  const location = useLocation();

  React.useEffect(() => {
    const checkMarketStatus = () => {
      try {
        const formatter = new Intl.DateTimeFormat('en-US', {
          timeZone: 'Asia/Kathmandu',
          hour: 'numeric',
          weekday: 'short',
          hour12: false
        });
        const parts = formatter.formatToParts(new Date());
        let hour = 0;
        let weekday = '';
        parts.forEach(p => {
          if (p.type === 'hour') hour = parseInt(p.value, 10);
          if (p.type === 'weekday') weekday = p.value;
        });
        
        if (weekday === 'Fri' || weekday === 'Sat' || hour < 11 || hour >= 15) {
          setIsMarketOpen(false);
        } else {
          setIsMarketOpen(true);
        }
      } catch (e) {
        console.error("Error formatting time:", e);
      }
    };
    checkMarketStatus();
    const interval = setInterval(checkMarketStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Market', path: '/market', icon: ChartLine },
    { name: 'Portfolio', path: '/portfolio', icon: Briefcase },
    { name: 'About', path: '/about', icon: Info },
  ];

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex overflow-hidden">
      {/* Sidebar */}
      <aside className={`bg-neutral-900 border-r border-neutral-800 transition-all duration-300 ease-in-out ${isSidebarOpen ? 'w-64' : 'w-20'} flex flex-col h-screen`}>
        <div className="p-6 flex items-center justify-between">
          {isSidebarOpen && <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-emerald-500 bg-clip-text text-transparent">NEPSE AI</h1>}
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-neutral-800 rounded-lg transition-colors">
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <nav className="flex-1 mt-6 px-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center p-3 rounded-xl transition-all duration-200 group ${
                  isActive 
                    ? 'bg-blue-600/10 text-blue-400 ring-1 ring-blue-600/20' 
                    : 'text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200'
                }`}
              >
                <Icon size={22} className={`${isActive ? 'text-blue-400' : 'group-hover:text-neutral-200'}`} />
                {isSidebarOpen && <span className="ml-4 font-medium">{item.name}</span>}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-neutral-800">
          <div className="flex items-center p-3 bg-neutral-800/50 rounded-xl">
             <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center font-bold text-sm">U</div>
             {isSidebarOpen && (
               <div className="ml-3">
                 <p className="text-sm font-medium">Guest User</p>
                 <p className="text-xs text-neutral-500">Free Account</p>
               </div>
             )}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-screen overflow-y-auto flex flex-col">
        <header className="h-16 border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-md sticky top-0 z-10 px-8 flex items-center justify-between shrink-0">
          <h2 className="text-lg font-semibold text-neutral-200">
            {navItems.find(i => i.path === location.pathname)?.name || 'Dashboard'}
          </h2>
          <div className="flex items-center space-x-4">
             <span className={`text-xs font-mono px-2 py-1 rounded-full border ${
               isMarketOpen 
                 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                 : 'bg-red-500/10 text-red-500 border-red-500/20'
             }`}>
               {isMarketOpen ? 'Market Open' : 'Market off'}
             </span>
             <button className="text-sm font-medium text-neutral-400 hover:text-neutral-100 transition-colors">
               Log in
             </button>
          </div>
        </header>

        <div className="p-8 flex-1">
          {children}
        </div>
        
        <Footer />
      </main>
    </div>
  );
};

export default Layout;
