const { MACD, RSI, SMA, EMA, BollingerBands, Stochastic, ATR, ADX, CCI, WilliamsR, OBV, MFI, PSAR } = require('technicalindicators');
const nepseService = require('./nepseService');

const analyzeSecurity = async (symbol) => {
    const [historyData, details] = await Promise.all([
        nepseService.getSecurityPriceVolumeHistory(symbol),
        nepseService.getSecurityDetails(symbol)
    ]);

    if (!historyData || !historyData.content || historyData.content.length === 0) {
        throw new Error('No historical timeline found for symbol');
    }
    
    const companyName = details?.security?.securityName || symbol;
    // Sort and validate data
    const items = historyData.content
        .filter(item => typeof item.closePrice === 'number' && item.businessDate)
        .sort((a, b) => new Date(a.businessDate) - new Date(b.businessDate));

    if (items.length < 20) throw new Error('Insufficient historical data for reliable analysis');

    const closePrices = items.map(i => i.closePrice);
    const highPrices  = items.map(i => i.highPrice || i.closePrice);
    const lowPrices   = items.map(i => i.lowPrice  || i.closePrice);
    const volumes     = items.map(i => i.totalTradedQuantity || 0);
    
    // ─── Core Indicators ────────────────────────────────────────────────────
    const rsiArr    = RSI.calculate({ period: 14, values: closePrices });
    const sma50Arr  = SMA.calculate({ period: 50, values: closePrices });
    const sma200Arr = SMA.calculate({ period: 200, values: closePrices });
    const ema9Arr   = EMA.calculate({ period: 9,  values: closePrices });
    const ema21Arr  = EMA.calculate({ period: 21, values: closePrices });
    const macdArr   = MACD.calculate({ 
        values: closePrices, 
        fastPeriod: 12, 
        slowPeriod: 26, 
        signalPeriod: 9, 
        SimpleMAOscillator: false, 
        SimpleMASignal: false 
    });
    const bbArr     = BollingerBands.calculate({ period: 20, values: closePrices, stdDev: 2 });
    const volSmaArr = SMA.calculate({ period: 20, values: volumes });

    // ─── Momentum Indicators ─────────────────────────────────────────────────
    const stochArr = Stochastic.calculate({
        high: highPrices, low: lowPrices, close: closePrices, period: 14, signalPeriod: 3
    });
    const cciArr      = CCI.calculate({ high: highPrices, low: lowPrices, close: closePrices, period: 20 });
    const williamrArr = WilliamsR.calculate({ high: highPrices, low: lowPrices, close: closePrices, period: 14 });
    const mfiArr      = MFI.calculate({ high: highPrices, low: lowPrices, close: closePrices, volume: volumes, period: 14 });

    // ─── Trend & Volatility Indicators ───────────────────────────────────────
    const atrArr  = ATR.calculate({ high: highPrices, low: lowPrices, close: closePrices, period: 14 });
    const adxArr  = ADX.calculate({ high: highPrices, low: lowPrices, close: closePrices, period: 14 });
    const psarArr = PSAR.calculate({ step: 0.02, max: 0.2, high: highPrices, low: lowPrices });
    const obvArr  = OBV.calculate({ close: closePrices, volume: volumes });

    // ─── Grab latest values ───────────────────────────────────────────────────
    const last = (arr) => arr.length > 0 ? arr[arr.length - 1] : null;
    const prev = (arr) => arr.length > 1 ? arr[arr.length - 2] : null;

    const lastPrice    = closePrices[closePrices.length - 1];
    const lastVolume   = volumes[volumes.length - 1];

    const currentRSI     = last(rsiArr);
    const currentSMA50   = last(sma50Arr);
    const currentSMA200  = last(sma200Arr);
    const currentEMA9    = last(ema9Arr);
    const currentEMA21   = last(ema21Arr);
    const currentMACD    = last(macdArr);
    const currentBB      = last(bbArr);
    const currentVolSMA  = last(volSmaArr) || 1;
    const currentStoch   = last(stochArr);
    const currentCCI     = last(cciArr);
    const currentWilliamR = last(williamrArr);
    const currentMFI     = last(mfiArr);
    const currentATR     = last(atrArr);
    const currentADX     = last(adxArr);
    const currentPSAR    = last(psarArr);
    const prevOBV        = prev(obvArr);
    const currentOBV     = last(obvArr);

    let signals = [];
    let score = 50; // Neutral start

    // ─── 1. RSI (±20) ────────────────────────────────────────────────────────
    if (currentRSI !== null) {
        if (currentRSI < 30)      { signals.push('RSI Oversold');         score += 20; }
        else if (currentRSI < 40) { signals.push('RSI Near Oversold');    score += 10; }
        else if (currentRSI > 70) { signals.push('RSI Overbought');       score -= 20; }
        else if (currentRSI > 60) { signals.push('RSI Near Overbought');  score -= 10; }
    }

    // ─── 2. SMA Price Action & Golden/Death Cross (±25) ──────────────────────
    if (currentSMA50) {
        if (lastPrice > currentSMA50) { signals.push('Price Above SMA50'); score += 10; }
        else                          { signals.push('Price Below SMA50'); score -= 10; }

        if (currentSMA200) {
            if (currentSMA50 > currentSMA200) { signals.push('Golden Cross (Structural Bullish)'); score += 15; }
            else                              { signals.push('Death Cross (Structural Bearish)');  score -= 15; }
        }
    }

    // ─── 3. EMA 9/21 Crossover (±10) ─────────────────────────────────────────
    if (currentEMA9 && currentEMA21) {
        if (currentEMA9 > currentEMA21) { signals.push('EMA9 > EMA21 (Short-term Bullish)'); score += 10; }
        else                            { signals.push('EMA9 < EMA21 (Short-term Bearish)'); score -= 10; }
    }

    // ─── 4. MACD (±15) ────────────────────────────────────────────────────────
    if (currentMACD?.MACD !== undefined && currentMACD?.signal !== undefined) {
        if (currentMACD.MACD > currentMACD.signal) { signals.push('MACD Positive Crossover'); score += 15; }
        else                                       { signals.push('MACD Negative Crossover'); score -= 15; }
        if (currentMACD.histogram > 0 && currentMACD.histogram > (prev(macdArr)?.histogram || 0))
            signals.push('MACD Histogram Rising');
    }

    // ─── 5. Bollinger Bands (±15) ──────────────────────────────────────────────
    if (currentBB) {
        if (lastPrice <= currentBB.lower)      { signals.push('BB Lower Touch (Potential Bounce)');   score += 15; }
        else if (lastPrice >= currentBB.upper) { signals.push('BB Upper Touch (Potential Pullback)'); score -= 15; }
        const bandwidth = (currentBB.upper - currentBB.lower) / currentBB.middle;
        if (bandwidth < 0.05) signals.push('BB Squeeze (Breakout Expected)');
    }

    // ─── 6. Stochastic (±15) ─────────────────────────────────────────────────
    if (currentStoch) {
        if (currentStoch.k < 20 && currentStoch.d < 20) { signals.push('Stochastic Oversold');  score += 15; }
        else if (currentStoch.k > 80 && currentStoch.d > 80) { signals.push('Stochastic Overbought'); score -= 15; }
        else if (currentStoch.k > currentStoch.d) signals.push('Stochastic Bullish Cross');
        else signals.push('Stochastic Bearish Cross');
    }

    // ─── 7. CCI (±10) ────────────────────────────────────────────────────────
    if (currentCCI !== null) {
        if (currentCCI < -100)     { signals.push('CCI Oversold (Bullish)');  score += 10; }
        else if (currentCCI > 100) { signals.push('CCI Overbought (Bearish)'); score -= 10; }
    }

    // ─── 8. Williams %R (±10) ─────────────────────────────────────────────────
    if (currentWilliamR !== null) {
        if (currentWilliamR < -80)      { signals.push('Williams %R Oversold');  score += 10; }
        else if (currentWilliamR > -20) { signals.push('Williams %R Overbought'); score -= 10; }
    }

    // ─── 9. MFI (±10) ─────────────────────────────────────────────────────────
    if (currentMFI !== null) {
        if (currentMFI < 20)      { signals.push('MFI Oversold (Money Outflow)'); score += 10; }
        else if (currentMFI > 80) { signals.push('MFI Overbought (Money Inflow)'); score -= 10; }
    }

    // ─── 10. PSAR (±10) ───────────────────────────────────────────────────────
    if (currentPSAR !== null) {
        if (lastPrice > currentPSAR) { signals.push('PSAR Bullish (Price Above SAR)'); score += 10; }
        else                         { signals.push('PSAR Bearish (Price Below SAR)'); score -= 10; }
    }

    // ─── 11. OBV Trend (context only) ─────────────────────────────────────────
    if (currentOBV !== null && prevOBV !== null) {
        if (currentOBV > prevOBV) signals.push('OBV Rising (Volume Confirms Uptrend)');
        else                      signals.push('OBV Falling (Volume Confirms Downtrend)');
    }

    // ─── 12. Volume Confirmation (±15) ────────────────────────────────────────
    if (lastVolume > currentVolSMA * 1.8) {
        signals.push('High Volume Confirmation');
        if (score > 55) score += 15;
        if (score < 45) score -= 15;
    }

    // ─── 13. ADX Trend Strength Multiplier ────────────────────────────────────
    let adxText = null;
    if (currentADX?.adx) {
        if (currentADX.adx > 40)     adxText = 'ADX Very Strong Trend';
        else if (currentADX.adx > 25) adxText = 'ADX Strong Trend';
        else if (currentADX.adx > 20) adxText = 'ADX Developing Trend';
        else                          adxText = 'ADX Weak/No Trend';
        signals.push(adxText);
        // Amplify score when trend is strong
        if (currentADX.adx > 25) {
            score = 50 + (score - 50) * 1.2;
        }
    }

    // Clamp
    score = Math.max(0, Math.min(100, Math.round(score)));

    // Signal action
    let recommendation = 'neutral';
    let action = 'HOLD';
    if (score >= 80)      { recommendation = 'strong_buy';  action = 'STRONG BUY'; }
    else if (score >= 65) { recommendation = 'buy';         action = 'BUY'; }
    else if (score <= 20) { recommendation = 'strong_sell'; action = 'STRONG SELL'; }
    else if (score <= 35) { recommendation = 'sell';        action = 'SELL'; }

    return {
        companyName,
        efficiency_score: score,
        signal_level: action,
        recommendation,
        trend:    score > 55 ? 'BULLISH' : (score < 45 ? 'BEARISH' : 'NEUTRAL'),
        strength: (score > 80 || score < 20) ? 'STRONG' : (score > 65 || score < 35) ? 'MODERATE' : 'WEAK',
        signals,
        lastPrice,
        indicators: {
            // Trend
            currentEMA9,
            currentEMA21,
            currentSMA50,
            currentSMA200,
            currentPSAR,
            currentADX: currentADX ? { adx: currentADX.adx, pdi: currentADX.pdi, mdi: currentADX.mdi } : null,
            // Momentum
            currentRSI,
            currentStoch,
            currentCCI,
            currentWilliamR,
            currentMFI,
            currentMACD,
            // Volatility
            currentBB,
            currentATR,
            // Volume
            currentOBV,
            lastVolume,
            currentVolSMA
        }
    };
};

module.exports = { analyzeSecurity };
