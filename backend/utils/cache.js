const NodeCache = require("node-cache");
const appCache = new NodeCache();

const set = (key, val, ttlMs = 0) => {
    // node-cache ttl is in seconds, so we convert from ms to seconds rounded up
    const ttlSeconds = Math.ceil(ttlMs / 1000);
    appCache.set(key, val, ttlSeconds);
};

const get = (key) => appCache.get(key);

const del = (key) => appCache.del(key);

const flush = () => appCache.flushAll();

module.exports = {
    set,
    get,
    del,
    flush
};
