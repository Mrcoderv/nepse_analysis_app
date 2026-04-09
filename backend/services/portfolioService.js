const fs = require('fs');
const path = require('path');
const nepseService = require('./nepseService');
const analysisService = require('./analysisService');

const dataFile = path.resolve(__dirname, '..', 'data', 'portfolio.json');
const dataDir = path.dirname(dataFile);

// Bootstrapping storage directory and file
const ensureDataExists = () => {
    try {
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
            console.log("Created directory:", dataDir);
        }
        if (!fs.existsSync(dataFile)) {
            fs.writeFileSync(dataFile, JSON.stringify([]), 'utf8');
            console.log("Created file:", dataFile);
        }
    } catch (err) {
        console.error("Error ensuring data storage exists:", err);
    }
};

// Initial run
ensureDataExists();

// Simple in-memory cache for analysis results (5 mins)
const analysisCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; 

const getAnalysisWithCache = async (symbol) => {
    const cached = analysisCache.get(symbol);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.data;
    }
    try {
        const analysis = await analysisService.analyzeSecurity(symbol);
        const details = await nepseService.getSecurityDetails(symbol);
        
        // Enrich analysis with latest traded price from company details
        if (details && details.securityDailyTradeDto) {
            analysis.lastPrice = details.securityDailyTradeDto.lastTradedPrice || details.securityDailyTradeDto.closePrice || analysis.lastPrice;
        }
        
        analysisCache.set(symbol, { data: analysis, timestamp: Date.now() });
        return analysis;
    } catch (e) {
        console.error(`Analysis failed for ${symbol}:`, e.message);
        return null;
    }
};

const getPortfolio = async () => {
    const data = JSON.parse(fs.readFileSync(dataFile));
    try {
        const liveMarket = await nepseService.getLiveMarket();
        const priceMap = {};
        
        // Cache mapping for extreme speed
        if (Array.isArray(liveMarket)) {
            liveMarket.forEach(item => {
                priceMap[item.symbol] = item.lastTradedPrice || item.closePrice;
            });
        }
        
        const enrichedData = await Promise.all(data.map(async (item) => {
            const analysis = await getAnalysisWithCache(item.symbol);
            
            // Use analysis.lastPrice as currentPrice, fallback to live market or buyPrice
            const currentPrice = analysis ? analysis.lastPrice : (priceMap[item.symbol] || item.buyPrice);
            
            const totalInvestment = Number(item.quantity) * Number(item.buyPrice);
            const currentValue = Number(item.quantity) * currentPrice;
            const profitLoss = currentValue - totalInvestment;
            const percentReturn = totalInvestment > 0 ? (profitLoss / totalInvestment) * 100 : 0;
            
            return {
                ...item,
                currentPrice,
                totalInvestment,
                currentValue,
                profitLoss,
                percentReturn: Number(percentReturn.toFixed(2)),
                signal: analysis ? analysis.signal_level : 'N/A',
                recommendation: analysis ? analysis.recommendation : 'neutral'
            };
        }));
        return enrichedData;
    } catch (e) {
        console.error("Failed to map live market for portfolio:", e.message);
        return data; // Return uncalculated if live market fails
    }
};

const addStock = async (symbol, quantity, buyPrice) => {
    const data = JSON.parse(fs.readFileSync(dataFile));
    const qtyNum = Number(quantity);
    const buyPriceNum = Number(buyPrice);
    
    if (isNaN(qtyNum) || isNaN(buyPriceNum)) {
        throw new Error("Invalid quantity or buy price");
    }
    
    const existing = data.find(item => item.symbol === symbol.toUpperCase());
    
    if (existing) {
        const totalQty = Number(existing.quantity) + qtyNum;
        existing.buyPrice = ((Number(existing.buyPrice) * Number(existing.quantity)) + (buyPriceNum * qtyNum)) / totalQty;
        existing.quantity = totalQty;
    } else {
        data.push({ id: Date.now().toString(), symbol: symbol.toUpperCase(), quantity: qtyNum, buyPrice: buyPriceNum });
    }
    
    fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
    return await getPortfolio();
};

const deleteStock = async (id) => {
    let data = JSON.parse(fs.readFileSync(dataFile));
    data = data.filter(item => item.id.toString() !== id.toString());
    fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
    return await getPortfolio();
};

const updateStock = async (id, quantity, buyPrice) => {
    let data = JSON.parse(fs.readFileSync(dataFile));
    const index = data.findIndex(item => item.id.toString() === id.toString());
    if (index !== -1) {
        data[index] = {
            ...data[index],
            quantity: Number(quantity),
            buyPrice: Number(buyPrice)
        };
        fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
    }
    return await getPortfolio();
};

module.exports = { getPortfolio, addStock, deleteStock, updateStock };
