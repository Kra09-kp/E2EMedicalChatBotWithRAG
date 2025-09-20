// Initialize all tooltips
const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl)
})

const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

// Connect to FastAPI websocket
const ws = new WebSocket("ws://localhost:8000/ws/ask");

ws.onopen = () => console.log("Connected to chatbot server.");
let botDiv = null;  // store the current bot message div
let waitingForBot = false;

// Append message bubbles
function addBubble(text, who = "user") {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${who}`;
    if (who === "user") {
        bubble.innerHTML = `<div class="bubble">${text}</div><div class="icon">🧑</div>`;
    } else {
        bubble.innerHTML = `<div class="icon">🤖</div><div class="bubble">${text}</div>`;
    }
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    }

// Typing indicator
function showTyping() {
    if (!document.getElementById("typingIndicator")) {
        const typing = document.createElement("div");
        typing.id = "typingIndicator";
        typing.className = "chat-bubble bot";
        typing.innerHTML = `<div class="icon">🤖</div>
        <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
        chatWindow.appendChild(typing);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

function removeTyping() {
    const t = document.getElementById("typingIndicator");
    if (t) t.remove();
}

function toggleSendLock(lock) {
    waitingForBot = lock;
    sendBtn.classList.toggle("disabled", lock);
}

// Send user message
// ... your previous setup code stays the same ...

let responseTimeout = null;  // timeout handler

function startResponseTimeout() {
    clearResponseTimeout(); // just in case
    responseTimeout = setTimeout(() => {
        removeTyping();
        addBubble("⚠️ Bot is taking too long. Please try again.", "bot");
        toggleSendLock(false);
        botDiv = null;
    }, 15000); // 15 seconds timeout
}

function clearResponseTimeout() {
    if (responseTimeout) {
        clearTimeout(responseTimeout);
        responseTimeout = null;
    }
}

// Send user message
function sendMessage() {
    if (waitingForBot) return;

    const text = chatInput.value.trim();
    if (!text) return;

    addBubble(text, "user");
    chatInput.value = "";

    toggleSendLock(true);
    showTyping();
    startResponseTimeout();  // start timeout

    // Send via websocket
    try {
        ws.send(JSON.stringify({ question: text }));
    } catch (err) {
        removeTyping();
        addBubble("⚠️ Error sending message. Please try again.", "bot");
        toggleSendLock(false);
        clearResponseTimeout();
    }
}
let markdownBuffer = "";
let renderTimer = null;
function startBotMessage() {
  // Create a new bot bubble for this message
  botDiv = document.createElement("div");
  botDiv.className = "chat-bubble bot";
  botDiv.innerHTML = `<div class="icon">🤖</div><div class="bubble"></div>`;
  chatWindow.appendChild(botDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // reset buffer for this new message
  markdownBuffer = "";
}


ws.onmessage = e => {
  const token = e.data;

  if (token === "[[END]]") {
    flushMarkdown();
    toggleSendLock(false);
    botDiv = null;
    clearResponseTimeout();
    return;
  }

  removeTyping();
  
  if (!botDiv) startBotMessage();   // start a new bot bubble if none exists
  markdownBuffer += token;


  // throttle parsing so we don’t run on every single character
  if (!renderTimer) {
    renderTimer = setTimeout(() => {
      flushMarkdown();
      renderTimer = null;
    }, 100);
  }

  startResponseTimeout();
};

function flushMarkdown() {
  if (!botDiv) return;
  const bubble = botDiv.querySelector(".bubble");
  bubble.innerHTML = marked.parse(markdownBuffer);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
// Handle websocket errors
ws.onerror = () => {
    removeTyping();
    addBubble("⚠️ WebSocket error. Try refreshing the page.", "bot");
    toggleSendLock(false);
    clearResponseTimeout();
};

ws.onclose = () => {
    removeTyping();
    console.log("⚠️ Connection closed");
    addBubble("Try again!","bot")
    toggleSendLock(false);
    clearResponseTimeout();
};


// Event listeners
sendBtn.addEventListener("click", () => sendMessage());
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
