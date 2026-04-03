const { MACD, RSI, SMA, BollingerBands } = require('technicalindicators');
const nepseService = require('./nepseService');

const analyzeSecurity = async (symbol) => {
    const historyData = await nepseService.getSecurityPriceVolumeHistory(symbol);
    if (!historyData || !historyData.content || historyData.content.length === 0) {
        throw new Error('No historical timeline found for symbol');
    }
    
    // Sort logically ascending by date using pure JS. 
    // Note: the backend natively uses date strings.
    const items = historyData.content
        .filter(item => typeof item.closePrice === 'number' && item.businessDate)
        .sort((a, b) => new Date(a.businessDate) - new Date(b.businessDate));

    if (items.length === 0) throw new Error('Insufficient validated history entries for Math');

    const closePrices = items.map(i => i.closePrice);
    
    const rsiArr = RSI.calculate({ period: 14, values: closePrices });
    const sma50Arr = SMA.calculate({ period: 50, values: closePrices });
    const sma200Arr = SMA.calculate({ period: 200, values: closePrices });
    const macdArr = MACD.calculate({ 
        values: closePrices, 
        fastPeriod: 12, 
        slowPeriod: 26, 
        signalPeriod: 9, 
        SimpleMAOscillator: false, 
        SimpleMASignal: false 
    });
    const bbArr = BollingerBands.calculate({ period: 20, values: closePrices, stdDev: 2 });
    
    // Grab realtime/latest slices
    const lastPrice = closePrices[closePrices.length - 1];
    const currentRSI = rsiArr.length > 0 ? rsiArr[rsiArr.length - 1] : null;
    const currentSMA50 = sma50Arr.length > 0 ? sma50Arr[sma50Arr.length - 1] : null;
    const currentSMA200 = sma200Arr.length > 0 ? sma200Arr[sma200Arr.length - 1] : null;
    const currentMACD = macdArr.length > 0 ? macdArr[macdArr.length - 1] : null;
    const currentBB = bbArr.length > 0 ? bbArr[bbArr.length - 1] : null;
    
    let trend = "NEUTRAL";
    let strength = "WEAK";
    let signals = [];

    // Sophisticated Signal Engine scoring rules processing
    let bullishScore = 0;
    let bearishScore = 0;

    if (currentRSI !== null) {
        if (currentRSI < 30) { signals.push('RSI Oversold (Bullish)'); bullishScore += 2; }
        else if (currentRSI > 70) { signals.push('RSI Overbought (Bearish)'); bearishScore += 2; }
    }

    if (currentSMA50 !== null) {
        if (lastPrice > currentSMA50) { signals.push('Price Above SMA50 (Bullish)'); bullishScore += 1; }
        else if (lastPrice < currentSMA50) { signals.push('Price Below SMA50 (Bearish)'); bearishScore += 1; }
        
        if (currentSMA200 !== null) {
            if (currentSMA50 > currentSMA200) { signals.push('Golden Cross structural trend (Bullish)'); bullishScore += 2; }
            else if (currentSMA50 < currentSMA200) { signals.push('Death Cross structural trend (Bearish)'); bearishScore += 2; }
        }
    }

    if (currentMACD !== null && currentMACD.MACD !== undefined && currentMACD.signal !== undefined) {
        if (currentMACD.MACD > currentMACD.signal) { signals.push('MACD Positive Crossover (Bullish)'); bullishScore += 2; }
        else if (currentMACD.MACD < currentMACD.signal) { signals.push('MACD Negative Crossover (Bearish)'); bearishScore += 2; }
    }

    if (bullishScore > bearishScore && bullishScore >= 3) {
        trend = "BULLISH";
        strength = bullishScore >= 5 ? "STRONG" : "MODERATE";
    } else if (bearishScore > bullishScore && bearishScore >= 3) {
        trend = "BEARISH";
        strength = bearishScore >= 5 ? "STRONG" : "MODERATE";
    }

    if (signals.length === 0) signals.push('No dominant signals detected.');

    return {
        trend,
        strength,
        signals,
        lastPrice,
        indicators: {
            currentRSI,
            currentSMA50,
            currentSMA200,
            currentMACD, 
            currentBB
        }
    };
};

module.exports = { analyzeSecurity };
