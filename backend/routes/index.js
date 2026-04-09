const express = require('express');
const router = express.Router();
const nepseService = require('../services/nepseService');

router.get('/LiveMarket', async (req, res, next) => {
    try {
        const data = await nepseService.getLiveMarket();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/TopGainers', async (req, res, next) => {
    try {
        const data = await nepseService.getTopTenGainers();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/TopLosers', async (req, res, next) => {
    try {
        const data = await nepseService.getTopTenLosers();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/NepseIndex', async (req, res, next) => {
    try {
        const data = await nepseService.getNepseIndex();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/MarketSummary', async (req, res, next) => {
    try {
        const data = await nepseService.getMarketSummary();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/CompanyList', async (req, res, next) => {
    try {
        const data = await nepseService.getCompanyList();
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/CompanyDetails/:symbol', async (req, res, next) => {
    try {
        const data = await nepseService.getSecurityDetails(req.params.symbol);
        res.json(data);
    } catch (err) { next(err); }
});

router.get('/history/:symbol', async (req, res, next) => {
    try {
        const data = await nepseService.getSecurityPriceVolumeHistory(req.params.symbol);
        res.json(data);
    } catch (err) {
        console.error(`Error fetching history for symbol ${req.params.symbol}:`, err);
        next(err);
    }
});

router.get('/floorsheet', async (req, res, next) => {
    try {
        const data = await nepseService.getFloorSheet();
        res.json(data);
    } catch (err) { next(err); }
});

const portfolioService = require('../services/portfolioService');
const analysisService = require('../services/analysisService');
const aiService = require('../services/aiService');

// Advanced Features:
// Portfolio Tracking Endpoints
router.get('/portfolio', async (req, res, next) => {
    try {
        const data = await portfolioService.getPortfolio();
        res.json(data);
    } catch (err) { next(err); }
});

router.post('/portfolio', async (req, res, next) => {
    try {
        const { symbol, quantity, buyPrice } = req.body;
        if (!symbol || !quantity || !buyPrice) {
            return res.status(400).json({ error: "Missing required fields: symbol, quantity, buyPrice" });
        }
        const data = await portfolioService.addStock(symbol, quantity, buyPrice);
        res.json(data);
    } catch (err) { next(err); }
});

router.put('/portfolio/:id', async (req, res, next) => {
    try {
        const { quantity, buyPrice } = req.body;
        if (!quantity || !buyPrice) {
            return res.status(400).json({ error: "Missing required fields: quantity, buyPrice" });
        }
        const data = await portfolioService.updateStock(req.params.id, quantity, buyPrice);
        res.json(data);
    } catch (err) { next(err); }
});

router.delete('/portfolio/:id', async (req, res, next) => {
    try {
        const data = await portfolioService.deleteStock(req.params.id);
        res.json(data);
    } catch (err) { next(err); }
});

// Advanced Stock Analysis Engine endpoint
router.get('/analyze/:symbol', async (req, res, next) => {
    try {
        const analysis = await analysisService.analyzeSecurity(req.params.symbol);
        res.json(analysis);
    } catch (err) {
        console.error(`Error analyzing symbol ${req.params.symbol}:`, err);
        next(err);
    }
});

router.get('/signals/:symbol', async (req, res, next) => {
    try {
        // Just route it to analyze which also has the signal engine integrated
        const analysis = await analysisService.analyzeSecurity(req.params.symbol);
        res.json({ trend: analysis.trend, strength: analysis.strength, signals: analysis.signals });
    } catch (err) { next(err); }
});

// AI endpoints
router.get('/ai-analysis/:symbol', async (req, res, next) => {
    try {
        const analysis = await aiService.analyzeWithAI(req.params.symbol);
        res.json(analysis);
    } catch (err) { next(err); }
});

module.exports = router;
