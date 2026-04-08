const nepseService = require('./services/nepseService');

async function test() {
    try {
        console.log('Testing nepseService.getSecurityPriceVolumeHistory for NICA...');
        const history = await nepseService.getSecurityPriceVolumeHistory('NICA');
        console.log('History keys:', Object.keys(history));
        if (history.content) {
            console.log('Content length:', history.content.length);
        } else {
            console.log('No content field in response');
        }
    } catch (e) {
        console.error('Test failed:', e.message);
    }
}
test();
