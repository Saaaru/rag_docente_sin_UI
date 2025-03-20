document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const chatHistory = document.getElementById("chat-history");
    const resetButton = document.getElementById("reset-button"); // Botón de reinicio

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }

    const threadId = getCookie("thread_id"); // Obtenemos el thread_id de la cookie

    // Función para agregar mensajes al historial
    function addMessageToChat(message, role, sources) {
        const messageContainer = document.createElement("div");
        messageContainer.classList.add("message-container");
        messageContainer.classList.add(
            role === "user" ? "user-message" : "bot-message"
        );

        const messageBubble = document.createElement("div");
        messageBubble.classList.add("message-bubble");

        const messageIcon = document.createElement("i");
        messageIcon.classList.add("fas", "message-icon");
        messageIcon.classList.add(role === "user" ? "fa-user" : "fa-robot");

        const messageText = document.createElement("p");
        messageText.textContent = message;

        messageBubble.appendChild(messageIcon);
        messageBubble.appendChild(messageText);
        messageContainer.appendChild(messageBubble);

        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.classList.add("message-sources");
            sourcesDiv.innerHTML = `<small>Fuentes: ${sources.join(", ")}</small>`;
            messageContainer.appendChild(sourcesDiv);
        }

        chatHistory.appendChild(messageContainer);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll
    }

    // Función para enviar mensajes al backend
    function sendMessage(message) {
        if (!message.trim()) return;

        addMessageToChat(message, "user"); // Mostrar mensaje del usuario

        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `user_message=${encodeURIComponent(message)}&session_id=${encodeURIComponent(threadId)}`,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text();
        })
        .then((html) => {
            // Insertar la respuesta HTML *directamente*
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            const agentResponseDiv = tempDiv.firstChild;

            // Extraer el mensaje y las fuentes del HTML
            const messageText = agentResponseDiv.querySelector('.message-bubble p')?.textContent;
            const sourcesText = agentResponseDiv.querySelector('.message-sources small')?.textContent;
            const sources = sourcesText ? sourcesText.replace('Fuentes: ', '').split(', ') : [];

            if (messageText) {
                addMessageToChat(messageText, "bot", sources); // Mostrar respuesta del bot
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            addMessageToChat("Hubo un error al procesar tu pregunta.", "bot");
        });

        userInput.value = ""; // Limpiar input
    }

    // Event listeners
    sendButton.addEventListener("click", () => sendMessage(userInput.value));
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage(userInput.value);
        }
    });

    // Botón de reinicio
    resetButton.addEventListener("click", () => {
        fetch("/reset", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `session_id=${encodeURIComponent(threadId)}`,
        })
        .then((response) => response.json())
        .then((data) => {
            if (data.message === "Conversación reiniciada") {
                chatHistory.innerHTML = ""; // Limpiar el historial
                addMessageToChat("¡Conversación reiniciada!", "bot");
            }
        })
        .catch((error) => console.error("Error:", error));
    });
});