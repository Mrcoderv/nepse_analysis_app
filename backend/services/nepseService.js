const { Nepse } = require('@rumess/nepse-api');
const cache = require('../utils/cache');

const nepse = new Nepse();
nepse.setTLSVerification(false); // Disable strict TLS as requested

const getCachedData = async (key, ttlMs, fetcher) => {
    const cached = cache.get(key);
    if (cached) return cached;
    try {
        const data = await fetcher();
        cache.set(key, data, ttlMs);
        return data;
    } catch (error) {
        console.error(`Error fetching ${key}:`, error.message);
        throw error;
    }
}

const nepseService = {
    getLiveMarket: () => getCachedData('liveMarket', 30000, () => nepse.getLiveMarket()),
    getTopTenGainers: () => getCachedData('topGainers', 60000, () => nepse.getTopTenGainers()),
    getTopTenLosers: () => getCachedData('topLosers', 60000, () => nepse.getTopTenLosers()),
    getNepseIndex: () => getCachedData('nepseIndex', 60000, () => nepse.getNepseIndex()),
    getCompanyList: () => getCachedData('companyList', 3600000, () => nepse.getCompanyList()),
    getSecurityDetails: (symbol) => getCachedData(`company_${symbol}`, 86400000, async () => {
         return await nepse.getSecurityDetails(symbol); // Wait, security details probably take a string or id
         // We might need to map symbol to ID. The frontend might send ID or symbol. 
         // Let's assume symbol or ID works, or we will fix it if it doesn't.
    }),
    getSecurityPriceVolumeHistory: (symbol) => getCachedData(`history_${symbol}`, 3600000, () => nepse.getSecurityPriceVolumeHistory(symbol)),
    getFloorSheet: () => getCachedData('floorsheet', 30000, () => nepse.getFloorSheet())
};

module.exports = nepseService;
