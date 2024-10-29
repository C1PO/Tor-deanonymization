const express = require('express');
const app = express();
const path = require('path');
const PORT = 3000;
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json()); 
app.get('/get-ip', (req, res) => {
    const userIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
    res.json({ ip: userIp });
});
app.post('/track-focus', (req, res) => {
    const { ip, timestamp, status } = req.body;
    console.log(`Event: ${status === 'focus' ? 'has the focus' : 'has the blur'} - IP: ${ip}, Timestamp: ${timestamp}`);
    res.status(200).send('Event take succesfully');
});
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server on: http://localhost:${PORT}`);
});
