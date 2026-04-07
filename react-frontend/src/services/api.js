import axios from 'axios';

const API_BASE_URL = 'https://nepse-analysis-app.onrender.com/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getLiveMarket = () => api.get('/LiveMarket');
export const getTopGainers = () => api.get('/TopGainers');
export const getTopLosers = () => api.get('/TopLosers');
export const getNepseIndex = () => api.get('/NepseIndex');
export const getCompanyList = () => api.get('/CompanyList');
export const getCompanyDetails = (symbol) => api.get(`/CompanyDetails/${symbol}`);
export const getStockHistory = (symbol) => api.get(`/history/${symbol}`);
export const getPortfolio = () => api.get('/portfolio');
export const addStockToPortfolio = (stockData) => api.post('/portfolio', stockData);
export const getStockAnalysis = (symbol) => api.get(`/analyze/${symbol}`);
export const getAIAnalysis = (symbol) => api.get(`/ai-analysis/${symbol}`);

export default api;
