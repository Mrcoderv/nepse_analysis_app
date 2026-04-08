const { MACD, RSI, SMA, BollingerBands } = require('technicalindicators');
const nepseService = require('./nepseService');

const analyzeSecurity = async (symbol) => {
    const historyData = await nepseService.getSecurityPriceVolumeHistory(symbol);
    if (!historyData || !historyData.content || historyData.content.length === 0) {
        throw new Error('No historical timeline found for symbol');
    }
    
    // Sort and validate data
    const items = historyData.content
        .filter(item => typeof item.closePrice === 'number' && item.businessDate)
        .sort((a, b) => new Date(a.businessDate) - new Date(b.businessDate));

    if (items.length < 20) throw new Error('Insufficient historical data for reliable analysis');

    const closePrices = items.map(i => i.closePrice);
    const volumes = items.map(i => i.totalTradedQuantity || 0);
    
    // Indicators
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
    
    // Volume SMA (20-day)
    const volSmaArr = SMA.calculate({ period: 20, values: volumes });

    // Current Values
    const lastPrice = closePrices[closePrices.length - 1];
    const lastVolume = volumes[volumes.length - 1];
    const currentRSI = rsiArr[rsiArr.length - 1];
    const currentSMA50 = sma50Arr[sma50Arr.length - 1] || null;
    const currentSMA200 = sma200Arr[sma200Arr.length - 1] || null;
    const currentMACD = macdArr[macdArr.length - 1] || null;
    const currentBB = bbArr[bbArr.length - 1] || null;
    const currentVolSMA = volSmaArr[volSmaArr.length - 1] || 0;
    
    let signals = [];
    let score = 50; // Start at Neutral

    // 1. RSI Analysis (Max +/- 20)
    if (currentRSI !== undefined) {
        if (currentRSI < 30) { signals.push('Oversold (Bullish)'); score += 20; }
        else if (currentRSI < 40) { signals.push('Near Oversold'); score += 10; }
        else if (currentRSI > 70) { signals.push('Overbought (Bearish)'); score -= 20; }
        else if (currentRSI > 60) { signals.push('Near Overbought'); score -= 10; }
    }

    // 2. SMA Crossovers & Price Action (Max +/- 25)
    if (currentSMA50) {
        if (lastPrice > currentSMA50) { signals.push('Price Above SMA50'); score += 10; }
        else { signals.push('Price Below SMA50'); score -= 10; }

        if (currentSMA200) {
            if (currentSMA50 > currentSMA200) { signals.push('Golden Cross (Structural Bullish)'); score += 15; }
            else { signals.push('Death Cross (Structural Bearish)'); score -= 15; }
        }
    }

    // 3. MACD Analysis (Max +/- 15)
    if (currentMACD && currentMACD.MACD !== undefined && currentMACD.signal !== undefined) {
        if (currentMACD.MACD > currentMACD.signal) { signals.push('MACD Positive Crossover'); score += 15; }
        else { signals.push('MACD Negative Crossover'); score -= 15; }
    }

    // 4. Bollinger Bands (Max +/- 15)
    if (currentBB) {
        if (lastPrice <= currentBB.lower) { signals.push('BB Lower Touch (Potential Bounce)'); score += 15; }
        else if (lastPrice >= currentBB.upper) { signals.push('BB Upper Touch (Potential Pullback)'); score -= 15; }
        
        // Squeeze Detection
        const bandwidth = (currentBB.upper - currentBB.lower) / currentBB.middle;
        if (bandwidth < 0.05) signals.push('BB Squeeze (Volatility Expected)');
    }

    // 5. Volume Confirmation (Multiplier effect or +/- 15)
    if (lastVolume > currentVolSMA * 1.8) {
        signals.push('High Volume Confirmation');
        if (score > 55) score += 15; // Confirm move up
        if (score < 45) score -= 15; // Confirm move down
    }

    // Wrap and Clamp score
    score = Math.max(0, Math.min(100, score));

    // Determine Action & Recommendation
    let recommendation = "neutral";
    let action = "HOLD";
    
    if (score >= 80) { recommendation = "strong_buy"; action = "STRONG BUY"; }
    else if (score >= 65) { recommendation = "buy"; action = "BUY"; }
    else if (score <= 20) { recommendation = "strong_sell"; action = "STRONG SELL"; }
    else if (score <= 35) { recommendation = "sell"; action = "SELL"; }

    return {
        efficiency_score: score,
        signal_level: action,
        recommendation,
        trend: score > 55 ? "BULLISH" : (score < 45 ? "BEARISH" : "NEUTRAL"),
        strength: (score > 80 || score < 20) ? "STRONG" : (score > 65 || score < 35) ? "MODERATE" : "WEAK",
        signals,
        lastPrice,
        indicators: {
            currentRSI,
            currentSMA50,
            currentSMA200,
            currentMACD, 
            currentBB,
            currentVolSMA,
            lastVolume
        }
    };
};

module.exports = { analyzeSecurity };
