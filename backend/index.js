require('dotenv').config();
const express = require('express');
const cors = require('cors');
const apiRoutes = require('./routes/index');

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.json());

app.use('/api', apiRoutes);
app.get('/', (req, res) => {
    res.send('Welcome to the NEPSE Analysis API!');
});

// Add a testing route
app.get('/test', (req, res) => {
    res.json({ message: 'Testing endpoint is working!' });
});

// Global Error Handler
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: err.message || 'Internal Server Error' });
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
