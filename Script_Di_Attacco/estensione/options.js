document.addEventListener('DOMContentLoaded', () => {
    const urlList = document.getElementById('urlList');
  
    // Recupera gli URL salvati con ID univoco
    browser.storage.local.get('urls').then((dati) => {
      const urls = dati.urls || [];
      urls.forEach((item) => {
        const li = document.createElement('li');
        li.textContent = `Session ID: ${item.sessionId} | URL: ${item.url} | Timestamp: ${item.timestamp}`;
        urlList.appendChild(li);
      });
    });
  });
  