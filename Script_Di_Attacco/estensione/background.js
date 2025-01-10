// Funzione per generare un ID univoco
function generateUUID() {
  return crypto.randomUUID();
}
const sessionId = generateUUID();
// Funzione per ottenere l'URL della scheda attiva
function ottieniURLAttivo() {
  browser.tabs.query({ active: true, currentWindow: true })
    .then((schede) => {
      if (schede.length > 0) {
        const url = schede[0].url;
        if (url.startsWith("about:")) {
          console.log("Pagina interna ignorata:", url);
          return;
        }
        console.log("URL attivo:", url);        
        
        // Salva l'URL e l'ID univoco nel local storage con timestamp
        browser.storage.local.get('urls')
          .then((dati) => {
            const urls = dati.urls || [];
            urls.push({ url: url, sessionId: sessionId, timestamp: new Date().toISOString() });
            return browser.storage.local.set({ urls });
          });
      }
    })
    .catch((errore) => console.error("Errore nell'ottenere l'URL attivo:", errore));
}

// Listener per il cambio di scheda
browser.tabs.onActivated.addListener(() => {
  ottieniURLAttivo();
});

// Listener per il caricamento o aggiornamento di una scheda
browser.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    ottieniURLAttivo();
  }
});
