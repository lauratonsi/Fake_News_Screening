// Right-click a selection on any page -> screen it via the API -> notify.
const API_URL = "http://127.0.0.1:8000/screen";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "screen-selection",
    title: "Screen “%s” for fake news",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "screen-selection" || !info.selectionText) return;
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: info.selectionText }),
    });
    const r = await res.json();
    const review = r.needs_review ? " — needs human review" : "";
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon128.png",
      title: `Screening: ${r.verdict} (${(r.fake_probability * 100).toFixed(0)}%, ${r.confidence})${review}`,
      message: r.disclaimer || "A screening aid, not a verdict.",
    });
  } catch (e) {
    chrome.notifications.create({
      type: "basic", iconUrl: "icon128.png",
      title: "Screening failed", message: String(e),
    });
  }
});
