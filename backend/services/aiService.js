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

    // Step 2: Build rigorous Gemini Prompt
    const prompt = `You are a professional NEPSE stock analyst.

Analyze this stock:

Symbol: ${symbol}
Price: ${data.lastPrice}
RSI: ${indicators.currentRSI}
MACD: ${indicators.currentMACD?.MACD}
Moving Average 50: ${indicators.currentSMA50}
Moving Average 200: ${indicators.currentSMA200}
Trend: ${data.trend}
Volume Profile: Not deeply provided

Tasks:
1. Determine market condition (Bullish/Bearish/Neutral)
2. Explain reasoning clearly
3. Predict short-term outlook
4. Assign risk level (Low/Medium/High)
5. Give recommendation (Buy/Sell/Hold)

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
