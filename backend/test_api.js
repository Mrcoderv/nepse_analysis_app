const { Nepse } = require('@rumess/nepse-api');
const nepse = new Nepse();
nepse.setTLSVerification(false);

async function test() {
    try {
        const history = await nepse.getSecurityPriceVolumeHistory('NICA');
        console.log('History keys:', Object.keys(history));
        if (history.content) {
            console.log('Content length:', history.content.length);
            console.log('First item:', history.content[0]);
        } else {
            console.log('No content field in response');
            console.log('Full response:', JSON.stringify(history).substring(0, 500));
        }
    } catch (e) {
        console.error(e);
    }
}
test();
