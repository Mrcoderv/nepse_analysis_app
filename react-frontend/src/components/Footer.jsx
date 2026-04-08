import React from 'react';
import { Earth, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="mt-auto py-8 px-8 border-t border-neutral-800 bg-neutral-950">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="text-neutral-500 text-sm">
          © {new Date().getFullYear()} <span className="text-neutral-300 font-medium">raghav vian panthi</span>. All rights reserved.
        </div>
        
        <div className="flex items-center space-x-6">
          <a 
            href="https://www.raghavpanthi.com.np/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-blue-400 transition-colors"
            title="Website"
          >
            <Earth size={20} />
          </a>
          <a 
            href="https://www.linkedin.com/in/raghav-vian-panthi/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-blue-500 transition-colors"
            title="LinkedIn"
          >
            <ExternalLink size={20} />
          </a>
          <a 
            href="https://www.youtube.com/@RaghavVian" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-neutral-500 hover:text-red-500 transition-colors"
            title="YouTube"
          >
            <ExternalLink size={20} />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
