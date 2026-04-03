const fs = require('fs');
const path = require('path');
const nepseService = require('./nepseService');

const dataDir = path.join(__dirname, '../data');
const dataFile = path.join(dataDir, 'portfolio.json');

// Bootstrapping storage directory and file
if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir);
}
if (!fs.existsSync(dataFile)) {
    fs.writeFileSync(dataFile, JSON.stringify([]));
}

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
        
        return data.map(item => {
            const currentPrice = priceMap[item.symbol] || item.buyPrice;
            const totalInvestment = item.quantity * item.buyPrice;
            const currentValue = item.quantity * currentPrice;
            const profitLoss = currentValue - totalInvestment;
            const percentReturn = totalInvestment > 0 ? (profitLoss / totalInvestment) * 100 : 0;
            
            return {
                ...item,
                currentPrice,
                totalInvestment,
                currentValue,
                profitLoss,
                percentReturn: Number(percentReturn.toFixed(2))
            };
        });
    } catch (e) {
        console.error("Failed to map live market for portfolio:", e.message);
        return data; // Return uncalculated if live market fails
    }
};

const addStock = async (symbol, quantity, buyPrice) => {
    const data = JSON.parse(fs.readFileSync(dataFile));
    const qtyNum = Number(quantity);
    const buyPriceNum = Number(buyPrice);
    
    const existing = data.find(item => item.symbol === symbol.toUpperCase());
    
    if (existing) {
        const totalQty = existing.quantity + qtyNum;
        existing.buyPrice = ((existing.buyPrice * existing.quantity) + (buyPriceNum * qtyNum)) / totalQty;
        existing.quantity = totalQty;
    } else {
        data.push({ id: Date.now().toString(), symbol: symbol.toUpperCase(), quantity: qtyNum, buyPrice: buyPriceNum });
    }
    
    fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
    return await getPortfolio();
};

module.exports = { getPortfolio, addStock };
