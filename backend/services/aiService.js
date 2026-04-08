const { GoogleGenerativeAI } = require('@google/generative-ai');
const analysisService = require('./analysisService');
const cache = require('../utils/cache');

// Initialize Gemini SDK safely
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "dummy_key");

const analyzeWithAI = async (symbol) => {
    // We cache AI responses to avoid repeated token costs and API calls
    const cacheKey = `ai_analysis_${symbol}`;
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("GEMINI_API_KEY is not configured in .env");
    }

    // Step 1: Precompute analytical baseline mathematically
    const data = await analysisService.analyzeSecurity(symbol);
    const indicators = data.indicators;

    // Step 2: Build rigorous Gemini Prompt with all indicators
    const ind = data.indicators;
    const prompt = `You are a professional NEPSE stock analyst with deep technical expertise.

Analyze this stock with the following comprehensive indicator data:

## Stock: ${symbol}
- **Current Price**: Rs. ${data.lastPrice}
- **Efficiency Score**: ${data.efficiency_score}/100
- **System Signal**: ${data.signal_level}

## Trend Indicators
- EMA 9: ${ind.currentEMA9?.toFixed(2) || 'N/A'}
- EMA 21: ${ind.currentEMA21?.toFixed(2) || 'N/A'}
- SMA 50: ${ind.currentSMA50?.toFixed(2) || 'N/A'}
- SMA 200: ${ind.currentSMA200?.toFixed(2) || 'N/A'}
- ADX: ${ind.currentADX?.adx?.toFixed(1) || 'N/A'} (PDI: ${ind.currentADX?.pdi?.toFixed(1)}, MDI: ${ind.currentADX?.mdi?.toFixed(1)})
- PSAR: ${ind.currentPSAR?.toFixed(2) || 'N/A'} (Price ${data.lastPrice > ind.currentPSAR ? 'ABOVE' : 'BELOW'} SAR)

## Momentum Indicators
- RSI (14): ${ind.currentRSI?.toFixed(2) || 'N/A'}
- Stochastic %K: ${ind.currentStoch?.k?.toFixed(1) || 'N/A'} / %D: ${ind.currentStoch?.d?.toFixed(1) || 'N/A'}
- CCI (20): ${ind.currentCCI?.toFixed(1) || 'N/A'}
- Williams %R: ${ind.currentWilliamR?.toFixed(1) || 'N/A'}
- MFI (14): ${ind.currentMFI?.toFixed(1) || 'N/A'}
- MACD: ${ind.currentMACD?.MACD?.toFixed(2)}, Signal: ${ind.currentMACD?.signal?.toFixed(2)}, Histogram: ${ind.currentMACD?.histogram?.toFixed(2)}

## Volatility Indicators
- ATR (14): ${ind.currentATR?.toFixed(2) || 'N/A'}
- Bollinger Bands — Upper: ${ind.currentBB?.upper?.toFixed(2)}, Middle: ${ind.currentBB?.middle?.toFixed(2)}, Lower: ${ind.currentBB?.lower?.toFixed(2)}
- BB Compression: ${ind.currentBB ? ((ind.currentBB.upper - ind.currentBB.lower) / ind.currentBB.middle * 100).toFixed(1) + '%' : 'N/A'}

## Volume Context
- Last Volume: ${ind.lastVolume?.toLocaleString() || 'N/A'}
- 20-day Volume SMA: ${ind.currentVolSMA ? Math.round(ind.currentVolSMA).toLocaleString() : 'N/A'}
- Volume Ratio: ${((ind.lastVolume || 0) / (ind.currentVolSMA || 1)).toFixed(2)}x average

## Active Signals
${data.signals.map(s => `- ${s}`).join('\n')}

---
Based on all the above data, provide your analysis:
1. Synthesize all indicators into a clear market assessment
2. Identify the strongest confirming/conflicting signals
3. Predict short-term (1-5 day) and medium-term (1-4 week) outlook
4. Quantify risk level for NEPSE market context
5. Give a precise recommendation with your reasoning

Return ONLY JSON:
{
  "trend": "",
  "strength": "",
  "reason": "",
  "outlook": "",
  "risk": "",
  "recommendation": ""
}`;

    const model = genAI.getGenerativeModel({ model: "gemini-flash-latest" });
    const result = await model.generateContent(prompt);
    let outputText = result.response.text();
    
    // Safety 1: Strip markdown JSON blocks if present
    outputText = outputText.replace(/```json/g, '').replace(/```/g, '').trim();

    try {
        const parsedJSON = JSON.parse(outputText);
        // Cache for 6 hours
        cache.set(cacheKey, parsedJSON, 21600000);
        return parsedJSON;
    } catch (e) {
        throw new Error(`Failed to parse AI response into strict JSON. Output was: ${outputText}`);
    }
};

module.exports = { analyzeWithAI };
